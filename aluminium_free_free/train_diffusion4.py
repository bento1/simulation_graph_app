import torch
import torch.nn.functional as F
from fem_dataset_to_image_full_mesh_diffusion import FemImageDataset as MeshDataset
from fem_model import UNet2D
import json
from torch.utils.data import DataLoader
from utils import EarlyStopping
import gc
import numpy as np
import matplotlib.pyplot as plt
from prediction_diffusion_image import sample_image

early_stopping = EarlyStopping(
    patience=50,     # FEM/GNN은 15~30 권장
    min_delta=1e-6,  # loss 스케일에 맞게
    mode="min"
)
def collate_flatten(batch): 
    images, conds = [], [] 
    batch_size=len(batch) 
    step_num_in_batch=len(batch[0]["image"]) 
    idxs=[] 
    for _ in range(batch_size): 
        idx = torch.randperm(len(batch[0]["image"])) 
        idxs.append(idx) 
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

class DiffusionScheduler:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.betas = torch.linspace(1e-4, 0.02, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

def q_sample(x0, t, sched, batch):
    """
    x0    : [N,Co]
    t     : [B]      (mesh-wise)
    batch : [N]
    """
    noise = torch.randn_like(x0)
    ab = sched.alpha_bar[t][batch]
    ab=ab.view(-1, 1, 1, 1)     
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
    return x_t, noise

def diffusion_loss(model, batch_data, sched):
    cond  = batch_data["cond"]
    x0    = batch_data["image"]

    B = batch_data['cond'].shape[0] 
    t = torch.randint(0, sched.T, (B,), device=x0.device)

    x_t, noise = q_sample(x0, t, sched, torch.Tensor([range(0,  B,)]).int())
    noise_pred = model(x_t, t/sched.T, cond,)

    return F.mse_loss(noise_pred, noise)

@torch.no_grad()
def sample_mesh(model, pos, cond, sched):
    """
    pos  : [N,3]   (새 mesh)
    cond : [N,Cc]
    """
    device = pos.device
    N, Co = pos.size(0), model.fc_out.out_features

    x = torch.randn(N, Co, device=device)
    batch = torch.zeros(N, dtype=torch.long, device=device)  # single mesh

    for t in reversed(range(sched.T)):
        t_tensor = torch.tensor([t], device=device)
        eps = model(pos, cond, x, t_tensor, batch)

        alpha = sched.alphas[t]
        ab = sched.alpha_bar[t]
        beta = sched.betas[t]

        x = (1 / torch.sqrt(alpha)) * (
            x - (1 - alpha) / torch.sqrt(1 - ab) * eps
        )

        if t > 0:
            x += torch.sqrt(beta) * torch.randn_like(x)

    return x

def draw(data1,data2,model_param): 
    stride=1 
    clip_q=(0.25,0.75) 
    cmap="jet" 
    ql, qh = np.quantile(data1.cpu().detach().numpy(), clip_q) 
    Uf1_clip = np.clip(data1.cpu().detach().numpy(), ql, qh) 
    ql, qh = np.quantile(data2.cpu().detach().numpy(), clip_q) 
    Uf2_clip = np.clip(data2.cpu().detach().numpy(), ql, qh) 
    plt.figure(figsize=(8, 4)) 
    plt.subplot(1, 2, 1) 
    plt.imshow(Uf1_clip, cmap="gray") 
    plt.title("Image 1") 
    plt.axis("off") 
    plt.subplot(1, 2, 2) 
    plt.imshow(Uf2_clip, cmap="gray")
    plt.title("Image 2") 
    plt.axis("off") 
    plt.tight_layout() 
    plt.show()

def train_one_epoch(model, loader, device, opt, sched, model_param): 
    model.train() 
    total = 0.0 
    count=0 
    for i, batch in enumerate(loader): 
        imags=batch['image'] 
        conds=batch['cond'] 
        for b in range(len(imags)): 
            batch = {'image':torch.stack(imags[b]).to(device),'cond':torch.Tensor(np.array(conds[b])).to(device)} 
            loss = diffusion_loss(model, batch, sched) 
            opt.zero_grad() 
            loss.backward() 
            opt.step() 
            total += float(loss.item()) 
            count+=1 
            if count % 100 == 0: 
                print(i,'batch',count,'step',float(loss.item())) 
                origin_image=torch.stack(imags[b]).to(device)[0,:][(model_param['in_channels']-1)//2]
                cond=torch.Tensor(np.array(conds[b])).to(device)[0,:] 
                pred_image = sample_image(model, cond, sched,(1,model_param['in_channels'],model_param['GRID'],model_param['GRID']), device) 
                pred_image=pred_image[0,:][(model_param['in_channels']-1)//2] 
                draw(origin_image,pred_image,model_param)
        del batch 
        gc.collect() 
    return total / max(1, count)

@torch.no_grad()
def eval_one_epoch(model, loader, device,  sched,model_param):
    model.eval()
    total = 0.0 
    count=0 
    for i, batch in enumerate(loader): 
        imags=batch['image'] 
        conds=batch['cond'] 
        for b in range(len(imags)): 
            batch = {'image':torch.stack(imags[b]).to(device),'cond':torch.Tensor(np.array(conds[b])).to(device)} 
            loss = diffusion_loss(model, batch, sched) 
            total += float(loss.item()) 
            count+=1 
            if count % 100 == 0: 
                print(i,'batch',count,'step',float(loss.item())) 
                origin_image=torch.stack(imags[b]).to(device)[0,:][(model_param['in_channels']-1)//2]
                cond=torch.Tensor(np.array(conds[b])).to(device)[0,:] 
                pred_image = sample_image(model, cond, sched,(1,model_param['in_channels'],model_param['GRID'],model_param['GRID']), device) 
                pred_image=pred_image[0,:][(model_param['in_channels']-1)//2] 
                draw(origin_image,pred_image,model_param)
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

    model_param={ 'in_channels':5, 
                'out_channels':5,
                'GRID':64,
                'loss_scale':1.0,
                'learning_rate':1e-4,
                'num_epochs':1000,
                'time_dim':256,
                'base':256,
                'dropout':0.01,
                'T':1000, } 
    
    train_ds = MeshDataset(root_dir=root,type='train',GRID=model_param['GRID'],in_channels=5) 
    val_ds= MeshDataset(root_dir=root,type='valid',GRID=model_param['GRID'],in_channels=5) 
    sched = DiffusionScheduler(device=device) 
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_flatten) 
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_flatten) 
    example = train_ds[0] 
    model_param['cond_dim']=len(example['cond'][0]) 
    model_param['dataset_scale_info']=train_ds.scale_info 
    model = UNet2D( in_channels=model_param['in_channels'], 
                   out_channels=model_param['out_channels'], 
                   cond_dim=model_param['cond_dim'], 
                   base=model_param['base'], 
                   time_dim=model_param['time_dim'], 
                   dropout=model_param['dropout'] ).to(device)
    sched.T=model_param['T']

    optimizer = torch.optim.Adam(model.parameters(), lr=model_param['learning_rate'], weight_decay=1e-6)
    loss_dict={}
    for epoch in range(model_param['num_epochs']):        
        train_loss = train_one_epoch(model, train_loader, device, optimizer, sched,model_param)
        val_loss = eval_one_epoch(model, val_loader, device, sched,model_param)
        loss_dict[epoch] = {'train_loss':train_loss, 'val_loss':val_loss}   
        print(f"Epoch {epoch+1}/{model_param['num_epochs']}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}")  
        improved = early_stopping.step(val_loss)
        with open(f"loss_history_diffustion_ver4.json", "w", encoding="utf-8") as f:
            json.dump(loss_dict, f, indent=2)
        if improved:
            best_val = val_loss
            # print(eval_one_epoch_sample_mse(model, val_loader, device, sched))
            torch.save(model.state_dict(), "mesh_invariant_diffustion_ver4_early.pt")  # best만 저장

            with open(f"model_param_diffustion_ver4_early.json", "w", encoding="utf-8") as f:
                json.dump(model_param, f, indent=2)
        if early_stopping.should_stop:
            print(
                f"\n Early stopping at epoch {epoch} "
                f"(best val = {best_val:.6e})"
            )
            break
        if epoch%10==0:
            torch.save(model.state_dict(), "mesh_invariant_diffustion_ver4.pt")
    torch.save(model.state_dict(), "mesh_invariant_diffustion_ver4.pt")
    with open(f"model_param_diffusion_ver4.json", "w", encoding="utf-8") as f:
        json.dump(model_param, f, indent=2) 

if __name__ == "__main__":
    main()