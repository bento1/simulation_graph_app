import torch
import torch.nn.functional as F
from fem_dataset_to_image_full_mesh_diffusion import FemImageDataset as MeshDataset
from fem_model import ContextUnet
import json
from torch.utils.data import DataLoader
from utils import EarlyStopping
import gc

early_stopping = EarlyStopping(
    patience=50,     # FEM/GNN은 15~30 권장
    min_delta=1e-6,  # loss 스케일에 맞게
    mode="min"
)

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

def train_one_epoch(model, loader, device, opt, sched):
    model.train()
    total = 0.0
    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        loss = diffusion_loss(model, batch, sched)

        opt.zero_grad()
        loss.backward()
        opt.step() 
        total += float(loss.item())
        if i % 10 == 0:
            print(i,'step',float(loss.item()))
        del batch
        gc.collect()
    return total / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, device,  sched):
    model.eval()
    total = 0.0
    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        loss = diffusion_loss(model, batch, sched)
        # batch = {k: v.to('cpu') for k, v in batch.items()}
        total += float(loss.item())
        # loss= loss.cpu()
        if i % 10 == 0:
            print(i,'step',float(loss.item()))
        del batch
        gc.collect()
    return total / max(1, len(loader))

def main():
    root = "./data"
    if torch.cuda.is_available() :
        device='cuda'
    elif torch.backends.mps.is_available() :
        device='mps'
    else :
        device='cpu'

    train_ds = MeshDataset(root_dir=root,type='train',GRID=128,in_channels=5)
    val_ds= MeshDataset(root_dir=root,type='valid',GRID=128,in_channels=5)
    sched = DiffusionScheduler(device=device)


    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    


    example = train_ds[0]

    model_param={
            'in_channels':5,
            'out_channels':5,
            'GRID':128,
            'dataset_scale_info':train_ds.scale_info,
            'loss_scale':1.0,
            'learning_rate':1e-4,
            'num_epochs':1000,
            'n_feat':256,
            'n_cfeat':example['cond'].shape[-1],
            'T':1000,
            }
    
    model = ContextUnet(
        in_channels=model_param['in_channels'],
        out_channels=model_param['out_channels'],
        n_feat=model_param['n_feat'],
        n_cfeat=model_param['n_cfeat'],
        height=model_param['GRID'],
        ).to(device)
    sched.T=model_param['T']
    optimizer = torch.optim.Adam(model.parameters(), lr=model_param['learning_rate'], weight_decay=1e-6)
    loss_dict={}
    for epoch in range(model_param['num_epochs']):        
        train_loss = train_one_epoch(model, train_loader, device, optimizer, sched)
        val_loss = eval_one_epoch(model, val_loader, device, sched)
        loss_dict[epoch] = {'train_loss':train_loss, 'val_loss':val_loss}   
        print(f"Epoch {epoch+1}/{model_param['num_epochs']}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}")  
        improved = early_stopping.step(val_loss)
        with open(f"loss_history_diffustion_ver3.json", "w", encoding="utf-8") as f:
            json.dump(loss_dict, f, indent=2)
        if improved:
            best_val = val_loss
            # print(eval_one_epoch_sample_mse(model, val_loader, device, sched))
            torch.save(model.state_dict(), "mesh_invariant_diffustion_ver3_early.pt")  # best만 저장

            with open(f"model_param_diffustion_ver3_early.json", "w", encoding="utf-8") as f:
                json.dump(model_param, f, indent=2)
        if early_stopping.should_stop:
            print(
                f"\n Early stopping at epoch {epoch} "
                f"(best val = {best_val:.6e})"
            )
            break
        if epoch%10==0:
            torch.save(model.state_dict(), "mesh_invariant_diffustion_ver3.pt")
    torch.save(model.state_dict(), "mesh_invariant_diffustion_ver3.pt")
    with open(f"model_param_diffusion_ver3.json", "w", encoding="utf-8") as f:
        json.dump(model_param, f, indent=2) 

if __name__ == "__main__":
    main()