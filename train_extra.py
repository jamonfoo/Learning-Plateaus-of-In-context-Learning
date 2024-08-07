import torch
from dataset.Imgdataset import ImgDataset
from models import TransformerModel
from torch import nn
# import wandb
import tqdm
import os
import argparse 
import yaml
import pickle
from utils.logger import setup_logger, get_timestamp
import numpy as np
_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}
def get_index(factors):

  indices = 0
  base = 1
  for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
    indices += factors[factor] * base
    base *= _NUM_VALUES_PER_FACTOR[name]
  return indices
def get_factors(index):  
    factors = [0] * len(_FACTORS_IN_ORDER)  
    base = 1  
    for factor, name in enumerate(_FACTORS_IN_ORDER):  
        base *= _NUM_VALUES_PER_FACTOR[name]  
    for factor, name in enumerate(_FACTORS_IN_ORDER):  
        base //= _NUM_VALUES_PER_FACTOR[name]  
        factors[factor] = index // base  
        index %= base  
    return factors 


sample_dir="dataset/sample"
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to input file')
    parser.add_argument('--output_dir', type=str, default="output", help='Path to output file')  
    parser.add_argument('--mode', type=str,default="context")
    args=parser.parse_args()
    output_dir=args.output_dir
    sample_path=os.path.join(sample_dir,f"train.npz")
    args = parser.parse_args()  
    with open(args.config, 'r') as yaml_file:  
        config= yaml.safe_load(yaml_file)  
    out_dir=os.path.join(output_dir,config["name"]+f"_{args.mode}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    device=config["device"]
    logpath=os.path.join(out_dir,"logs")
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    logger = setup_logger(config["name"], logpath,
            filename="{}.txt".format(get_timestamp()))
    logger.info({
                    "learning rate": config["lr"],
                    "n_head":config["n_head"],
                    "n_embd":config["n_embd"],
                    "n_layer":config["n_layer"],
                    "n_positions":config["n_positions"]
                })
    if args.mode=="context":
        trainset=ImgDataset(sample_path,return_factor=True)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)
        model=TransformerModel(n_dims=768, out_dim=768, n_positions=config["n_positions"], n_embd=config["n_embd"], n_layer=config["n_layer"], n_head=config["n_head"]).to(device)
        if "ckpt" in config:
            state_dict=torch.load(config["ckpt"])
            model.load_state_dict(state_dict,strict=False)
            print("load ckpt")

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=config["lr"])
        total_loss=0
        total_acc=0
        for epoch in range(config["num_epoch"]):
            for i,(imgs,labels,factors) in tqdm.tqdm(enumerate(trainloader),total=len(trainloader)):
                imgs,labels=imgs.to(device),labels.to(device)
                factors=torch.tensor(np.eye(imgs.shape[2])[factors.numpy()]).to(device).float()
                # extend the factors to the same length as the labels
                factors=factors.unsqueeze(1).repeat(1,labels.shape[1],1)
                

                optim.zero_grad()
                emb=model(imgs,labels)
                output=emb[:,:,:128]
                loss_1=criterion(output.reshape(-1,output.shape[-1]),torch.argmax(labels,dim=2).view(-1))
                factors_pred=emb[:,:,128:]
                loss_2=criterion(factors_pred.reshape(-1,factors_pred.shape[-1]),torch.argmax(factors,dim=2).view(-1))
                # loss_2=criterion(output.reshape(-1,output.shape[-1]),torch.argmax(factors,dim=2).view(-1))
                loss=loss_1+0.1*loss_2
                loss.backward()
                optim.step()
                acc=torch.mean((torch.argmax(labels[:,:-1,:],dim=2).view(-1)==torch.argmax(output[:,:-1,:],dim=2).view(-1)).float())
                total_acc+=acc
                total_loss+=loss.detach().item()
                if (len(trainloader)*epoch+i)%config["log_every_steps"]==0:
                    logger.info("Epoch: {}, Step: {}, Loss: {}, Acc: {}".format(epoch,len(trainloader)*epoch+i,total_loss/config["log_every_steps"],total_acc/config["log_every_steps"]))
                    total_loss=0
                    total_acc=0
            if epoch%config["ckpt_every_epoch"]==0:
                torch.save(model.state_dict(), os.path.join(out_dir, f"model_{epoch}.pt"))
    elif args.mode=="weights":
        
        trainset=ImgDataset(sample_path,return_index=True)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)
        model=TransformerModel(n_dims=768, out_dim=768, n_positions=config["n_positions"], n_embd=config["n_embd"], n_layer=config["n_layer"], n_head=config["n_head"]).to(device)

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=config["lr"])
        total_loss=0
        total_acc=0
        for epoch in range(config["num_epoch"]):
            for i,(imgs,labels,idxs) in tqdm.tqdm(enumerate(trainloader),total=len(trainloader)):
                imgs,labels=imgs.to(device),labels.to(device)
                factors=get_factors(idxs)
                factors=torch.stack(factors).permute(1, 2, 0).to(device)
                optim.zero_grad()
                emb=model(imgs,labels)
                output=emb[:,:,:128]
                loss_1=criterion(output.reshape(-1,output.shape[-1]),torch.argmax(labels,dim=2).view(-1))

                values=emb[:,:,128:128+15*6]
                
                values=values.reshape(-1,15)
                factors=factors.reshape(-1).long()
                loss_2=criterion(values,factors)
                loss=loss_1+0.1*loss_2
                loss.backward()
                optim.step()
                acc=torch.mean((torch.argmax(labels[:,:-1,:],dim=2).view(-1)==torch.argmax(output[:,:-1,:128],dim=2).view(-1)).float())
                total_acc+=acc
                total_loss+=loss.detach().item()
                if (len(trainloader)*epoch+i)%config["log_every_steps"]==0:
                    logger.info("Epoch: {}, Step: {}, Loss: {}, Acc: {}".format(epoch,len(trainloader)*epoch+i,total_loss/config["log_every_steps"],total_acc/config["log_every_steps"]))
                    total_loss=0
                    total_acc=0
            if epoch%config["ckpt_every_epoch"]==0:
                torch.save(model.state_dict(), os.path.join(out_dir, f"model_{epoch}.pt"))