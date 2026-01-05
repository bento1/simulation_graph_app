import torch
import torch.nn.functional as F
from fem_dataset_standard_scaler_full_mesh_diffusion import FemGraphDataset as MeshDataset
from fem_model import MeshInvariantDiffusion_ver2
import json
from torch.utils.data import DataLoader
from utils import EarlyStopping
import gc

early_stopping = EarlyStopping(
    patience=50,     # FEM/GNN은 15~30 권장
    min_delta=1e-6,  # loss 스케일에 맞게
    mode="min"
)
def collate_mesh_batch(batch):
    pos   = torch.cat([b["pos"]  for b in batch], dim=0)
    cond  = torch.cat([b["cond"] for b in batch], dim=0)
    x0    = torch.cat([b["x0"]   for b in batch], dim=0)

    batch_idx = torch.cat([
        torch.full((b["pos"].size(0),), i, dtype=torch.long)
        for i, b in enumerate(batch)
    ])

    return {
        "pos": pos,         # [ΣNi,3]
        "cond": cond,       # [ΣNi,Cc]
        "x0": x0,           # [ΣNi,Co]
        "batch": batch_idx  # [ΣNi]
    }

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
    ab = sched.alpha_bar[t][batch].unsqueeze(-1)

    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
    return x_t, noise

def diffusion_loss(model, batch_data, sched):
    pos   = batch_data["pos"]
    cond  = batch_data["cond"]
    x0    = batch_data["x0"]
    batch = batch_data["batch"]

    B = batch.max().item() + 1
    t = torch.randint(0, sched.T, (B,), device=x0.device)

    x_t, noise = q_sample(x0, t, sched, batch)
    noise_pred = model(pos, cond, x_t, t, batch)

    return F.mse_loss(noise_pred, noise)

def split_batch_by_mesh(batch):
    """
    batch:
      pos   [N,3]
      cond  [N,Cc]
      x0    [N,Co]
      batch [N]
    """
    meshes = []
    batch_idx = batch["batch"]

    for m in batch_idx.unique():
        mask = batch_idx == m
        meshes.append({
            "pos":  batch["pos"][mask],
            "cond": batch["cond"][mask],
            "x0":   batch["x0"][mask],
        })
    return meshes


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
def eval_one_epoch_sample_mse(model, loader, device, sched):
    model.eval()
    total_mse = 0.0
    count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # batch → mesh 단위로 분리
        meshes = split_batch_by_mesh(batch)

        for m in meshes:
            pos  = m["pos"]
            cond = m["cond"]
            x0   = m["x0"]

            x_hat = sample_mesh(model, pos, cond, sched)

            mse = F.mse_loss(x_hat, x0)

            total_mse += float(mse.item())
            count += 1
        if count==3:
            break
    return total_mse / max(1, count)

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

    train_ds = MeshDataset(root_dir=root,type='train')
    val_ds= MeshDataset(root_dir=root,type='valid')
    sched = DiffusionScheduler(device=device)


    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,collate_fn=collate_mesh_batch)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,collate_fn=collate_mesh_batch)
    


    example = train_ds[0]

    model_param={
            'pos_dim':3,
            'cond_dim':example['cond'].shape[-1],
            'out_dim':1,
            'dataset_scale_info':train_ds.scale_info,
            'loss_scale':1.0,
            'learning_rate':1e-4,
            'num_epochs':1000,
            'hidden':1024,
            'T':1000,
            'fourier_L': 32,
            'num_blocks':12
            }
    
    model = MeshInvariantDiffusion_ver2(
        pos_dim=model_param['pos_dim'],
        cond_dim=model_param['cond_dim'],
        out_dim=model_param['out_dim'],
        hidden=model_param['hidden'],
        T= model_param['T'],
        fourier_L=model_param['fourier_L'],
        num_blocks=model_param['num_blocks']
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model_param['learning_rate'], weight_decay=1e-6)
    loss_dict={}
    for epoch in range(model_param['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, sched)
        val_loss = eval_one_epoch(model, val_loader, device, sched)
        loss_dict[epoch] = {'train_loss':train_loss, 'val_loss':val_loss}   
        print(f"Epoch {epoch+1}/{model_param['num_epochs']}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}")  
        improved = early_stopping.step(val_loss)
        with open(f"loss_history_diffustion_ver2.json", "w", encoding="utf-8") as f:
            json.dump(loss_dict, f, indent=2)
        if improved:
            best_val = val_loss
            # print(eval_one_epoch_sample_mse(model, val_loader, device, sched))
            torch.save(model.state_dict(), "mesh_invariant_diffustion_ver2_early.pt")  # best만 저장

            with open(f"model_param_diffustion_ver2_early.json", "w", encoding="utf-8") as f:
                json.dump(model_param, f, indent=2)
        if early_stopping.should_stop:
            print(
                f"\n Early stopping at epoch {epoch} "
                f"(best val = {best_val:.6e})"
            )
            break
    torch.save(model.state_dict(), "mesh_invariant_diffustion_ver2.pt")

    with open(f"model_param_diffusion_ver2.json", "w", encoding="utf-8") as f:
        json.dump(model_param, f, indent=2) 

if __name__ == "__main__":
    main()