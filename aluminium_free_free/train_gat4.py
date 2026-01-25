# gnn_fem_mesh_invariant.py
import os, json
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from fem_dataset_standard_scaler import FemGraphDataset
from fem_model import MeshGNN_GAT4
from tqdm import tqdm
from utils import EarlyStopping
from torch_geometric.data import Data
import gc 
import random

def collate_flatten(batch): 
    batch_size=len(batch) 
    new_batch=[]
    for b in batch:
        new_batch.extend(b)
    del batch
    gc.collect()
    result=[]
    random.shuffle(new_batch)
    for i in range(0,len(new_batch)//batch_size,batch_size): 
        end_index = min(i+batch_size, len(new_batch))
        mini_batches=new_batch[i:end_index]
        mini_x=[]
        mini_y=[]
        mini_pos=[]
        mini_edge_index=[]
        mini_edge_attr=[]

        for mb in mini_batches:
            mini_x.append(mb.x)
            mini_y.append(mb.y)
            mini_pos.append(mb.pos)
            mini_edge_index.append(mb.edge_index)
            mini_edge_attr.append(mb.edge_attr)
        data = Data(
                x=torch.cat(mini_x, dim=0)  ,
                y=torch.cat(mini_y, dim=0)  ,
                pos=torch.cat(mini_pos, dim=0)  ,   # pos는 따로 보관(편함)
                edge_index=torch.cat(mini_edge_index, dim=1)  , 
                edge_attr=torch.cat(mini_edge_attr, dim=0)  ,
            )
        result.append(data)
    return result

early_stopping = EarlyStopping(
    patience=50,     # FEM/GNN은 15~30 권장
    min_delta=1e-8,  # loss 스케일에 맞게
    mode="min"
)
eps = 1e-8
best_val = float("inf")
def train_one_epoch(model, loader, opt, device, loss_scale):
    model.train()
    total = 0.0
    for i,batch in tqdm(enumerate(loader)):
        for mini_batch in batch:
            mini_batch = mini_batch.to(device)
            pred = model(mini_batch)
            loss = torch.sqrt(F.mse_loss(pred, mini_batch.y) + eps)*loss_scale
            del mini_batch, pred
            gc.collect()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, device,loss_scale):
    model.eval()
    total = 0.0
    for batch in loader:
        for mini_batch in batch:
            mini_batch = mini_batch.to(device)
            pred = model(mini_batch)
            loss = torch.sqrt(F.mse_loss(pred, mini_batch.y) + eps)*loss_scale
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

    train_ds = FemGraphDataset(root_dir=root,type='train', use_cell_edges=True)
    val_ds = FemGraphDataset(root_dir=root,type='valid', use_cell_edges=True)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,collate_fn=collate_flatten)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,collate_fn=collate_flatten)

    example = train_ds[0][0]

    model_param={'in_dim':example.x.shape[1],
            'edge_dim':4,
            'hidden':1024,
            'layers':16,
            'head':8,
            'out_dim':3,
            'dropout':0.001,
            'dataset_scale_info':train_ds.scale_info,
            'loss_scale':1.0,
            'learning_rate':1e-6
            }
    
    model = MeshGNN_GAT4(in_dim=model_param['in_dim'],
                edge_dim=model_param['edge_dim'],
                hidden=model_param['hidden'],
                layers=model_param['layers'],
                heads=model_param['head'],
                out_dim=model_param['out_dim'],
                dropout=model_param['dropout']).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=model_param['learning_rate'], weight_decay=1e-6)
    # if os.path.exists(f"mesh_invariant_gat4_early.pt"):
    #     checkpoint = torch.load(f"mesh_invariant_gat4_early.pt", map_location=device)
    #     model.load_state_dict(checkpoint)
    loss_dict={}
    for epoch in tqdm(range(1, 1000)):
        tr = train_one_epoch(model, train_loader, opt, device,model_param['loss_scale'])
        va = eval_one_epoch(model, val_loader, device,model_param['loss_scale'])
        loss_dict[epoch] = {'train_loss':tr, 'val_loss':va}   
        if epoch % 5 == 0 :
            print(f"epoch {epoch:03d} | train {tr:.6e} | val {va:.6e}")
        # early stopping 체크
        improved = early_stopping.step(va)

        if improved:
            best_val = va
            torch.save(model.state_dict(), "mesh_invariant_gat4_early.pt")  # best만 저장
            with open(f"loss_history_gat4_early.json", "w", encoding="utf-8") as f:
                json.dump(loss_dict, f, indent=2)
            with open(f"model_param_gat4_early.json", "w", encoding="utf-8") as f:
                json.dump(model_param, f, indent=2)
        if early_stopping.should_stop:
            print(
                f"\n Early stopping at epoch {epoch} "
                f"(best val = {best_val:.6e})"
            )
            break
    torch.save(model.state_dict(), "mesh_invariant_gat4.pt")
    print("saved: mesh_invariant_gat4.pt")
    with open(f"loss_history_gat4.json", "w", encoding="utf-8") as f:
        json.dump(loss_dict, f, indent=2)
    with open(f"model_param_gat4.json", "w", encoding="utf-8") as f:
        json.dump(model_param, f, indent=2)
if __name__ == "__main__":
    main()