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
sample_dir="dataset/sample"
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to input file')
    parser.add_argument('--output_dir', type=str, default="/home/t-jingwenfu/jwfu/in-context-learning/diff_incontext/output", help='Path to output file')  
    parser.add_argument('--num_maps', type=int)
    parser.add_argument('--num_try', type=int, default=0)
    args=parser.parse_args()
    output_dir=args.output_dir
    sample_path=os.path.join(sample_dir,f"train_{args.num_maps}.npz")
    args = parser.parse_args()  
    with open(args.config, 'r') as yaml_file:  
        config= yaml.safe_load(yaml_file)  
    out_dir=os.path.join(output_dir,config["name"]+f"_{args.num_maps}","try"+str(args.num_try))
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
    trainset=ImgDataset(sample_path)
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
        for i,(imgs,labels) in tqdm.tqdm(enumerate(trainloader),total=len(trainloader)):
            imgs,labels=imgs.to(device),labels.to(device)
            optim.zero_grad()

            output=model(imgs,labels)

            loss=criterion(output.view(-1,output.shape[-1]),torch.argmax(labels,dim=2).view(-1))
            loss.backward()
            optim.step()

            acc=torch.mean((torch.argmax(labels,dim=2).view(-1)==torch.argmax(output,dim=2).view(-1)).float())
            total_acc+=acc
            total_loss+=loss.detach().item()
            if (len(trainloader)*epoch+i)%config["log_every_steps"]==0:
                logger.info("Epoch: {}, Step: {}, Loss: {}, Acc: {}".format(epoch,len(trainloader)*epoch+i,total_loss/config["log_every_steps"],total_acc/config["log_every_steps"]))
                total_loss=0
                total_acc=0
        if epoch%config["ckpt_every_epoch"]==0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"model_{epoch}.pt"))
        
    