import torch
import torch.nn.functional as F
from fem_dataset_to_image_full_mesh_diffusion import FemImageDataset as MeshDataset
from fem_model import TinyLatentDiffusion,DDPMScheduler,ScalingVAE4,reparameterize
import json
from torch.utils.data import DataLoader
from utils import EarlyStopping, minmax_scale
import gc
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch
from collections import defaultdict


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


def drawHist(mu,std):
    plt.figure(figsize=(8, 4)) 
    plt.subplot(1, 2, 1) 
    plt.hist(mu.flatten(), bins=100) 
    plt.subplot(1, 2, 2) 
    plt.hist(std.flatten(), bins=100) 
    plt.title("Image 1") 
    plt.show()

def train_one_epoch(loader, device,VAE_model,): 

    total = 0.0 
    count=0 
    mean_result=[]
    std_result=[]
    for i, batch in enumerate(loader): 
        imags=batch['image'] 
        conds=batch['cond'] 
        for b in range(len(imags)): 
            s_batch = {'image':torch.stack(imags[b]).to(device),'cond':torch.Tensor(np.array(conds[b])).to(device)} 
            mean = s_batch['image'].mean(dim=(1,2,3), keepdim=True).detach() 
            std = s_batch['image'].std(dim=(1,2,3), keepdim=True).detach() + 1e-12 
            mean_result.append(mean.cpu().detach().numpy().reshape((-1,1)))
            std_result.append(std.cpu().detach().numpy().reshape((-1,1)))
            # x_norm=(s_batch['image']-mean)/std 
            # mu, logvar = VAE_model.encoder(x_norm) 
            # z = reparameterize(mu, logvar) 
            # B,C,X,Y=z.shape 
            # z = torch.cat([z, mean.expand(B,1,X,Y), std.expand(B,1,X,Y)], dim=1) # [B, Z+2] 

    mean_result=np.vstack(mean_result)
    std_result=np.vstack(std_result)
    drawHist(mean_result,np.log(std_result**2))

    drawHist((mean_result-mean_result.mean())/mean_result.std(),(np.log(std_result**2)-np.log(std_result**2).mean())/(np.log(std_result**2).std()))
    mean_sacler_min=((mean_result-mean_result.mean())/mean_result.std()).min()
    mean_sacler_max=((mean_result-mean_result.mean())/mean_result.std()).max()
    std_scaler_min=((np.log(std_result**2)-np.log(std_result**2).mean())/(np.log(std_result**2).std())).min()
    std_scaler_max=((np.log(std_result**2)-np.log(std_result**2).mean())/(np.log(std_result**2).std())).max()
    scaled_mean=(mean_result-mean_result.mean())/mean_result.std()
    scaled_std=(np.log(std_result**2)-np.log(std_result**2).mean())/(np.log(std_result**2).std())

    drawHist((mean_result-mean_result.mean())/mean_result.std(),(np.log(std_result**2)-np.log(std_result**2).mean())/(np.log(std_result**2).std()))
    drawHist(minmax_scale(scaled_mean,mean_sacler_min,mean_sacler_max,s=(-1,1)),minmax_scale(scaled_std,std_scaler_min,std_scaler_max,s=(-1,1)))
    return mean_result,std_result


def main():
    root = "./data"
    if torch.cuda.is_available() :
        device='cuda'
    elif torch.backends.mps.is_available() :
        device='mps'
    else :
        device='cpu'

    model_param={
                'dataset_in_channels':3,  
                'in_channels':256, 
                'out_channels':3,
                'GRID':64,
                'loss_scale':1.0,
                'learning_rate':5e-4,
                'num_epochs':1000,
                'base':256,
                'num_cond_tokens':128,
                'token_dim':128,
                'time_dim':128,
                'batch_size':128,
                'T':1000 } 
    with open(os.path.join( "model_param_diffustion_vae4_2_early.json"), "r", encoding="utf-8") as f:
        VAE_model_param = json.load(f)

    train_ds = MeshDataset(root_dir=root,type='train',GRID=model_param['GRID'],in_channels=model_param['dataset_in_channels']) 
    train_loader = DataLoader(train_ds, batch_size=model_param['batch_size'], shuffle=True, collate_fn=collate_flatten) 

    example = train_ds[0] 
    model_param['cond_dim']=len(example['cond'][0])
    model_param['dataset_scale_info']=train_ds.scale_info 

    
    VAE_model = ScalingVAE4( in_channels=VAE_model_param['in_channels'], 
                latent_dim=VAE_model_param['latent_dim'], 
                base=VAE_model_param['base'],
                depth=VAE_model_param['depth'] ).to(device)
    print('[LOAD[START] VAE param')
    ckpt = torch.load("mesh_invariant_diffustion_vae4_2_early_epoch019.pt", map_location=device)
    VAE_model.load_state_dict(ckpt)
    
    for p in tqdm(VAE_model.parameters()):
        p.requires_grad = False
    print('[LOAD][COMPLETE] VAE param')
    

    train_loss = train_one_epoch(train_loader, device,VAE_model)


if __name__ == "__main__":
    main()