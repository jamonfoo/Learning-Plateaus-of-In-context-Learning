import torch
from dataset.Imgdataset import ImgDataset
from models import TransformerModel,ProbeClassification
from torch import nn
import wandb
import tqdm
import os
import argparse 
import yaml
# import matplotlib.pyplot as plt
import tqdm
import pickle
import numpy as np
import json
sample_dir="dataset/sample"
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,default="config/standard.yaml", help='Path to input file')  
parser.add_argument('--output_dir', type=str, default="output", help='Path to output file')
parser.add_argument('--num_maps', type=int)
parser.add_argument('--num_try', type=int, default=0)
args = parser.parse_args()  
with open(args.config, 'r') as yaml_file:  
    config= yaml.safe_load(yaml_file)  
device=config["device"]
output_dir=args.output_dir

sample_path=os.path.join(sample_dir,f"test_{args.num_maps}.npz")

dir_path=os.path.join(output_dir,config["name"]+f"_{args.num_maps}","try"+str(args.num_try))
k_epoch=1
T_epoch=100
model=TransformerModel(n_dims=768, out_dim=768, n_positions=config["n_positions"], n_embd=config["n_embd"], n_layer=config["n_layer"], n_head=config["n_head"]).to(device)
criterion = nn.CrossEntropyLoss()
testset=ImgDataset(sample_path)
testloader = torch.utils.data.DataLoader(
testset, batch_size=128, shuffle=True, num_workers=2)
if __name__=="__main__":
    # evaluate in-context learning performance
    rst=[]
    path_list=os.listdir(dir_path)
    for p in tqdm.tqdm(path_list,total=len(path_list)):
        if p.endswith(".pt"):
            train_epoch=int(p.split("_")[1].split(".")[0])
            path=os.path.join(dir_path,p)
        else:
            continue
        if train_epoch%k_epoch !=0 or train_epoch>T_epoch:
            continue

        model.load_state_dict(torch.load(path),strict=False)
        total_acc=torch.zeros((config["n_positions"]))
        for i,(imgs,labels) in enumerate(testloader):
            imgs,labels=imgs.to(device),labels.to(device)
            output=model(imgs,labels)
            acc=torch.sum((torch.argmax(labels,dim=2)==torch.argmax(output,dim=2)).float(),dim=0)
            total_acc+=acc.cpu()
        total_acc/=len(testset)
        rst.append((train_epoch,total_acc.numpy().tolist()))
    json.dump(rst,open(os.path.join(dir_path,f"test_{test_type}.json"),"w"))    