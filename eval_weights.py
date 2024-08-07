import torch
from dataset.Imgdataset import ImgDataset
from models import TransformerModel,ProbeClassification
from torch import nn
import tqdm
import os
import argparse 
import yaml
import tqdm
import pickle
import numpy as np
import sys
import json

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

# def disable_attention(model):  
#     for layer in model.transformer.h:  
#         layer.attn.forward = lambda *args, **kwargs: (args[0], None)  
sample_dir="/home/t-jingwenfu/jwfu/in-context-learning/diff_incontext/dataset/sample"
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
# sample_path=os.path.join(sample_dir,f"test.npz")
test_type="D2D"
if test_type=="D2D":
    sample_path=os.path.join(sample_dir,f"test_{args.num_maps}.npz")
else:
    sample_path=os.path.join(sample_dir,f"test.npz")
dir_path=os.path.join(output_dir,config["name"]+f"_{args.num_maps}","try"+str(args.num_try))
k_epoch=1
T_epoch=100
model=TransformerModel(n_dims=768, out_dim=768, n_positions=config["n_positions"], n_embd=config["n_embd"], n_layer=config["n_layer"], n_head=config["n_head"]).to(device)
criterion = nn.CrossEntropyLoss()

trainset=ImgDataset(os.path.join(sample_dir,f"train_{args.num_maps}.npz"),return_index=True)
trainloader = torch.utils.data.DataLoader(
trainset, batch_size=128, shuffle=True, num_workers=2)

testset=ImgDataset(sample_path,return_index=True)
testloader = torch.utils.data.DataLoader(
testset, batch_size=128, shuffle=True, num_workers=2)


def obtain_model(register_layer):
    model=TransformerModel(n_dims=768, out_dim=768, n_positions=config["n_positions"], n_embd=config["n_embd"], n_layer=config["n_layer"], n_head=config["n_head"]).to(device)
    model.register_hooks(register_layer)
    for layer in model._backbone.h:  
        layer.attn.forward = lambda *args, **kwargs: (args[0], None)
    return model

if __name__=="__main__":
    # calculate in-weight representation
    

    file_path=os.path.join(dir_path,"model_49.pt")
    if not os.path.exists(file_path):
        print("no file: ",file_path)
        sys.exit()
    
    model=obtain_model(2).to(device)

    
    
    model.load_state_dict(torch.load(file_path))
        # model.eval()
    probemodel=ProbeClassification(probe_class=15, num_task=6,input_dim=128,linear=True).to(device)
    optim = torch.optim.Adam(probemodel.parameters(), lr=1e-3)
        
    for batch_idx,(xs,ys,idxs) in enumerate(trainloader):
        xs,ys=xs.to(device),ys.to(device)
        factors=get_factors(idxs)
        factors=torch.stack(factors).permute(1, 2, 0).to(device)
        # factors=torch.tensor(factors).to(device).view(-1)
        with torch.no_grad():
            out=model(xs,ys)
            embeds=model.intermediate_features[0][:, ::2, :]
        optim.zero_grad()
        logits,loss=probemodel(embeds.reshape((-1,embeds.shape[-1])),factors.reshape(-1))
        loss.backward()
        optim.step()
    total_acc=0
    total_loss=0
    for batch_idx,(xs,ys,idxs) in enumerate(testloader):
        xs,ys=xs.to(device),ys.to(device)
        factors=get_factors(idxs)
        factors=torch.stack(factors).permute(1, 2, 0).to(device)
        with torch.no_grad():
            out=model(xs,ys)
            embeds=model.intermediate_features[0][:, ::2, :]
            # print(embeds.shape)
            # print(factors.shape)
            logits,loss=probemodel(embeds.reshape((-1,embeds.shape[-1])),factors.reshape(-1))
        

        total_acc+=(logits[:,:,:].argmax(dim=2).view(-1)==factors.reshape(-1)).float().mean().item()
        # print(total_acc/(batch_idx+1))
        total_loss+=loss.detach().item()
                # if batch_idx%100==0:
    save_path=os.path.join(dir_path,"weights_49_cut.json")
    json.dump({"acc":total_acc/(batch_idx+1),"loss":total_loss/(batch_idx+1)},open(save_path,"w"))
    print({"acc":total_acc/(batch_idx+1),"loss":total_loss/(batch_idx+1)})
        
   