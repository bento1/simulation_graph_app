import torch
import torch.nn.functional as F
from fem_dataset_to_image_full_mesh_diffusion import FemImageDataset as MeshDataset
from fem_model import ScalingVAE5
import json
from torch.utils.data import DataLoader
from utils import EarlyStopping
import gc
import numpy as np
import matplotlib.pyplot as plt


early_stopping = EarlyStopping(
    patience=10,     # FEM/GNN은 15~30 권장
    min_delta=1e-6,  # loss 스케일에 맞게
    mode="min"
)
def collate_flatten(batch): 
    images, conds = [], [] 
    batch_size=len(batch) 
    step_num_in_batch=len(batch[0]["image"]) 
    idxs=[] 
    for i in range(batch_size): 
        idx = torch.randperm(len(batch[0]["image"])) 
        idxs.append(idx) 
        # print(batch[i]['min_max'])
    idxs=np.array(idxs).T 

    for i in range(step_num_in_batch): 
        batch_images=[] 
        batch_conds=[] 
        for b_i,b in enumerate(batch): 
            id=idxs[i][b_i] 
            batch_images.append(b['image'][id]) 
            batch_conds.append(b['cond'][id]) 
        images.append(batch_images) 
        conds.append(batch_conds) 
    return { "image": images, "cond": conds, }

def draw(data1,data1_1,data2,data3,model_param): 
    
    plt.figure(figsize=(8, 4)) 
    plt.subplot(1, 5, 1) 
    plt.imshow(data1.cpu().detach().numpy(), cmap="gray",vmin=data1.cpu().detach().numpy().min(),vmax=data1.cpu().detach().numpy().max()) 
    plt.title("Original Target") 
    plt.axis("off")
    plt.subplot(1, 5, 2) 
    plt.imshow(data1_1.cpu().detach().numpy(), cmap="gray",vmin=data1_1.cpu().detach().numpy().min(),vmax=data1_1.cpu().detach().numpy().max()) 
    plt.title("Original Scaled") 
    plt.axis("off") 
    plt.subplot(1, 5, 3) 
    plt.imshow(data2.cpu().detach().numpy(), cmap="gray",vmin=data1.cpu().detach().numpy().min(),vmax=data1.cpu().detach().numpy().max())
    plt.title("Predicted Image") 
    plt.axis("off") 
    plt.subplot(1, 5, 4) 
    plt.imshow(data3.cpu().detach().numpy(), cmap="gray",vmin=data1_1.cpu().detach().numpy().min(),vmax=data1_1.cpu().detach().numpy().max())
    plt.title("Predicted Scaled") 
    plt.axis("off")
    plt.subplot(1, 5, 5) 
    plt.imshow(np.abs(data1.cpu().detach().numpy()-data2.cpu().detach().numpy()), cmap="gray")
    plt.title("residual") 
    plt.axis("off")
    plt.tight_layout() 
    plt.show()

def kl_divergence_gaussian(mu, logvar, clamp=(-0.3, 0.3)):
    """
    KL( N(mu, var) || N(0, I) )
    Supports mu/logvar with shape:
      [B, Z] or [B, C, H, W] or [B, ...]
    """
    if clamp is not None:
        logvar = torch.clamp(logvar, min=clamp[0], max=clamp[1])

    # flatten all non-batch dimensions
    mu = mu.view(mu.size(0), -1)
    logvar = logvar.view(logvar.size(0), -1)

    kl = 0.5 * torch.sum(
        torch.exp(logvar) + mu**2 - 1.0 - logvar,
        dim=1
    )

    return kl.mean()

def charbonnier_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     eps: float = 1e-8,
                     reduction: str = "mean") -> torch.Tensor:
    """
    Charbonnier loss (a smooth L1 / pseudo-Huber variant):
        L = sqrt((pred - target)^2 + eps^2)

    Args:
        pred, target: same shape tensors
        eps: smoothing term (bigger -> more L1-like near 0)
        reduction: "mean" | "sum" | "none"

    Returns:
        loss tensor (scalar if reduced)
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred{pred.shape} vs target{target.shape}")

    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be one of {'mean','sum','none'}")




def GradientDifferenceLossCharbonnier( pred, target):
    grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    grad_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]

    grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    grad_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]

    return (
        charbonnier_loss(grad_pred_x, grad_target_x) +
        charbonnier_loss(grad_pred_y, grad_target_y)
    )
# x_norm,s_batch['image'], scaled_image,iamge,mean,std, image_mean,image_var,mu, logvar,
def cvae_loss_scale_only(target_x_norm,target,scaled_x_hat, x_hat,mean,std, mu, logvar,epoch, beta=1e-3): 
    recon_detail = charbonnier_loss(target_x_norm, scaled_x_hat)
    recon_detail_2 = GradientDifferenceLossCharbonnier(target_x_norm,scaled_x_hat)

    kl = kl_divergence_gaussian(mu, logvar)
    #return (0.8-0.4*epoch/100)*recon_detail +0.05*recon_detail_2+0.1*latent_mean+(0.1+0.4*epoch/100)*latent_logvar+ beta * kl, (recon_detail ,latent_mean,latent_logvar, kl)
    #return (0.8)*recon_detail +0.05*recon_detail_2+0.1*latent_mean+(0.1)*latent_logvar+ beta * kl, (recon_detail ,latent_mean,latent_logvar, kl)
    return 0.8*recon_detail +0.2*recon_detail_2 + beta * kl, (recon_detail , kl)

def cvae_loss(target_x_norm,target,scaled_x_hat, x_hat,mean,std, mu, logvar,epoch, beta=1e-3): 
    recon= charbonnier_loss(target, x_hat)
    recon_2 = GradientDifferenceLossCharbonnier(target,x_hat)
    recon_detail = charbonnier_loss(target_x_norm, scaled_x_hat)
    recon_detail_2 = GradientDifferenceLossCharbonnier(target_x_norm,scaled_x_hat)
    recon_sum=0.8*recon +0.2*recon_2
    recon_detail_sum=0.8*recon_detail +0.2*recon_detail_2
    
    kl = kl_divergence_gaussian(mu, logvar)
    #return (0.8-0.4*epoch/100)*recon_detail +0.05*recon_detail_2+0.1*latent_mean+(0.1+0.4*epoch/100)*latent_logvar+ beta * kl, (recon_detail ,latent_mean,latent_logvar, kl)
    return recon_sum + recon_detail_sum + beta * kl, (recon_sum,recon_detail_sum, kl)

def draw_latent(z): 

    z = z.view(z.size(0), -1) 
    plt.figure(figsize=(8, 4)) 
    plt.hist(z.flatten().detach().cpu(), bins=100) 
    plt.title("Image 1") 
    plt.show()

def train_one_epoch(model, loader, device, opt, model_param,epoch): 
    model.train() 
    total = 0.0 
    count=0 
    beta_target=1e-3
    warmup_epochs=10 
    warm = min(1.0, (epoch + 1) / max(1, warmup_epochs)) 
    beta = beta_target * warm 
    for i, batch in enumerate(loader): 
        imags=batch['image'] 
        conds=batch['cond'] 
        for b in range(len(imags)): 
            s_batch = {'image':torch.stack(imags[b]).to(device),'cond':torch.Tensor(np.array(conds[b])).to(device)} 
            mean = s_batch['image'].mean(dim=(1,2,3), keepdim=True).detach() 
            std = s_batch['image'].std(dim=(1,2,3), keepdim=True).detach() 
            x_norm=(s_batch['image']-mean)/std 
            std=torch.log(std**2+1e-12)
            scaled_image,image, mu, logvar, z= model(s_batch['image']) 
            loss, parts = cvae_loss(x_norm,s_batch['image'], scaled_image,image,mean,std,mu, logvar,epoch, beta=beta) 
            opt.zero_grad() 
            loss.backward() 
            opt.step() 
            total += float(loss.item())
            count+=1 
            if count % 50 == 0: 
                print('[TRAIN] epoch: ',epoch,'batch: ',i,'step: ',count,'loss: ',float(loss.item()),'recon_detail loss: ', parts) 
                origin_image=torch.stack(imags[b])[0,:][(model_param['in_channels']-1)//2] 
                print('BEFROE TRAIN')
                pred_image=image[0,:][(model_param['in_channels']-1)//2].cpu()
                origin_image_norm=x_norm[0,:][(model_param['in_channels']-1)//2].cpu()
                draw(origin_image,origin_image_norm,pred_image,scaled_image[0,:][(model_param['in_channels']-1)//2].cpu(),model_param) 
        del batch 
        gc.collect() 
    return total / max(1, count)
@torch.no_grad()
def eval_one_epoch(model, loader, device,  model_param,epoch):
    model.eval()
    total = 0.0 
    count=0 
    beta_target=1e-3
    warmup_epochs=10 
    warm = min(1.0, (epoch + 1) / max(1, warmup_epochs)) 
    beta = beta_target * warm 
    for i, batch in enumerate(loader): 
        imags=batch['image'] 
        conds=batch['cond'] 
        for b in range(len(imags)): 
            s_batch = {'image':torch.stack(imags[b]).to(device),'cond':torch.Tensor(np.array(conds[b])).to(device)} 
            mean = s_batch['image'].mean(dim=(1,2,3), keepdim=True).detach() 
            std = s_batch['image'].std(dim=(1,2,3), keepdim=True).detach() 
            x_norm=(s_batch['image']-mean)/std 
            std=torch.log(std**2+1e-12)
            scaled_image,image, mu, logvar, z= model(s_batch['image']) 
            loss, parts = cvae_loss(x_norm,s_batch['image'], scaled_image,image,mean,std,mu, logvar,epoch, beta=beta) 
            total += float(loss.item())
            count+=1 
            if count % 20 == 0: 
                print('[VALID] epoch: ',epoch,'batch: ',i,'step: ',count,'loss: ',float(loss.item()),'recon_detail loss: ', parts) 
                origin_image=torch.stack(imags[b])[0,:][(model_param['in_channels']-1)//2] 
                pred_image=image[0,:][(model_param['in_channels']-1)//2].cpu()
                origin_image_norm=x_norm[0,:][(model_param['in_channels']-1)//2].cpu()
                draw(origin_image,origin_image_norm,pred_image,scaled_image[0,:][(model_param['in_channels']-1)//2].cpu(),model_param) 
        del batch 
        gc.collect() 
    return total / max(1, count)

def main():
    root = "./data"
    if torch.cuda.is_available() :
        device='cuda'
    elif torch.backends.mps.is_available() :
        device='mps'
    else :
        device='cpu'

    model_param={ 'in_channels':3, 
                'out_channels':3,
                'GRID':64,
                'loss_scale':1.0,
                'learning_rate':1e-3,
                'num_epochs':1000,
                'base':128,
                'latent_dim':64,
                'batch_size':128,
                'depth':4 } 
    
    train_ds = MeshDataset(root_dir=root,type='train',GRID=model_param['GRID'],in_channels=model_param['in_channels']) 
    val_ds= MeshDataset(root_dir=root,type='valid',GRID=model_param['GRID'],in_channels=model_param['in_channels']) 

    train_loader = DataLoader(train_ds, batch_size=model_param['batch_size'], shuffle=True, collate_fn=collate_flatten) 
    val_loader = DataLoader(val_ds, batch_size=model_param['batch_size'], shuffle=False, collate_fn=collate_flatten) 
    example = train_ds[0] 
    model_param['cond_dim']=len(example['cond'][0]) 
    model_param['dataset_scale_info']=train_ds.scale_info 

    model = ScalingVAE5( in_channels=model_param['in_channels'], 
                latent_dim=model_param['latent_dim'], 
                base=model_param['base'],
                depth=model_param['depth'] ).to(device)
    # ckpt = torch.load("mesh_invariant_diffustion_vae4_4_early_epoch002.pt", map_location=device)
    # model.load_state_dict(ckpt)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_param['learning_rate'], weight_decay=1e-6)
    loss_dict={}
    for epoch in range(model_param['num_epochs']):        
        train_loss = train_one_epoch(model, train_loader, device, optimizer,model_param,epoch)
        val_loss = eval_one_epoch(model, val_loader, device,model_param,epoch)
        loss_dict[epoch] = {'train_loss':train_loss, 'val_loss':val_loss}   
        print(f"Epoch {epoch+1}/{model_param['num_epochs']}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}")  
        improved = early_stopping.step(val_loss)
        with open(f"loss_history_diffustion_vae4_3.json", "w", encoding="utf-8") as f:
            json.dump(loss_dict, f, indent=2)
        if improved:
            best_val = val_loss
            # print(eval_one_epoch_sample_mse(model, val_loader, device, sched))
            torch.save(model.state_dict(), f"mesh_invariant_diffustion_vae4_3_early_epoch{str(epoch).zfill(3)}.pt")  # best만 저장

            with open(f"model_param_diffustion_vae4_3_early.json", "w", encoding="utf-8") as f:
                json.dump(model_param, f, indent=2)
        if early_stopping.should_stop:
            print(
                f"\n Early stopping at epoch {epoch} "
                f"(best val = {best_val:.6e})"
            )
            break
        if epoch%5==0:
            torch.save(model.state_dict(), f"mesh_invariant_diffustion_vae4_3_epoch{str(epoch).zfill(3)}.pt")
    torch.save(model.state_dict(), "mesh_invariant_diffustion_vae4_3.pt")
    with open(f"model_param_diffusion_vae4_3.json", "w", encoding="utf-8") as f:
        json.dump(model_param, f, indent=2) 

if __name__ == "__main__":
    main()