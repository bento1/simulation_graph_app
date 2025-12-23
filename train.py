# gnn_fem_mesh_invariant.py
import os, json
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from torch_geometric.loader import DataLoader
from fem_dataset import FemGraphDataset
from fem_model import MeshGNN
from tqdm import tqdm
from utils import EarlyStopping
early_stopping = EarlyStopping(
    patience=50,     # FEM/GNN은 15~30 권장
    min_delta=1e-6,  # loss 스케일에 맞게
    mode="min"
)

best_val = float("inf")
def train_one_epoch(model, loader, opt, device, scale_y=1.0):
    model.train()
    total = 0.0
    for i,batch in enumerate(loader):
        batch.y = batch.y * scale_y
        batch = batch.to(device)
        pred = model(batch)
        mask = batch.core_mask
        loss = F.mse_loss(pred[mask], batch.y[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, device, scale_y=1.0):
    model.eval()
    total = 0.0
    for batch in loader:
        batch.y = batch.y * scale_y
        batch = batch.to(device)
        pred = model(batch)
        mask = batch.core_mask
        loss = F.mse_loss(pred[mask], batch.y[mask] )
        total += float(loss.item())
    return total / max(1, len(loader))


def main():
    root = "./data"
    if torch.cuda.is_available() :
        device='cuda'
    elif torch.backends.mps.is_available() :
        device='mps'
    else :
        device='cpu'

    ds = FemGraphDataset(root_dir=root, knn_k=12, use_cell_edges=True)

    n = len(ds)
    n_train = int(n * 0.8)
    train_ds = ds[:n_train]
    val_ds = ds[n_train:]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    example = ds[0]
    in_dim = example.x.shape[1]
    model_param={'in_dim':example.x.shape[1], 'edgie_dim':4, 'hidden':64, 'layers':4, 'out_dim':3, 'dropout':0.1,'SCALE_Y':1e5}
    model = MeshGNN(in_dim=in_dim, edge_dim=4, hidden=64, layers=4, out_dim=3, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

    # 변위 스케일(안정성용): e-6 수준이면 1e6 곱해서 학습 추천
    SCALE_Y = 1e5
    loss_dict={}
    for epoch in tqdm(range(1, 1000)):
        tr = train_one_epoch(model, train_loader, opt, device, scale_y=SCALE_Y)
        va = eval_one_epoch(model, val_loader, device, scale_y=SCALE_Y)
        loss_dict[epoch] = {'train_loss':tr, 'val_loss':va}   
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} | train {tr:.6e} | val {va:.6e}")
        # early stopping 체크
        improved = early_stopping.step(va)

        if improved:
            best_val = va
            torch.save(model.state_dict(), "mesh_invariant_gnn_early.pt")  # best만 저장
            with open(f"loss_history_early.json", "w", encoding="utf-8") as f:
                json.dump(loss_dict, f, indent=2)
            with open(f"model_param_early.json", "w", encoding="utf-8") as f:
                json.dump(model_param, f, indent=2)
        if early_stopping.should_stop:
            print(
                f"\n🛑 Early stopping at epoch {epoch} "
                f"(best val = {best_val:.6e})"
            )
            break
    torch.save(model.state_dict(), "mesh_invariant_gnn.pt")
    print("saved: mesh_invariant_gnn.pt")
    with open(f"loss_history.json", "w", encoding="utf-8") as f:
        json.dump(loss_dict, f, indent=2)
    with open(f"model_param.json", "w", encoding="utf-8") as f:
        json.dump(model_param, f, indent=2)
if __name__ == "__main__":
    main()