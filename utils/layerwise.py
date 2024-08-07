import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
dir_path="/home/v-jingwenfu/jwfu2/in-context-learning/img_incontext/output/standard_save/"
all_weights=[]
preweights=0
for file in os.listdir(dir_path):
    epoch=int(file.split("_")[-1].split(".")[0])
    # if epoch%5!=0:
    #     continue
    path=os.path.join(dir_path,file)
    state=torch.load(path)
    weights=[]
    for name,weight in state.items():
        if len(weight.shape)>1 and weight.dtype!=torch.bool:
            # print(name,weight.dtype)
            weights.append((name,weight.cpu().numpy()))
    if preweights==0:
        preweights=copy.deepcopy(weights)
    else:
        all_weights.append((epoch,[(weights[i][0],np.linalg.norm(weights[i][1]-preweights[i][1])) for i in range(len(weights))]))

matrix=[[weight[1] for weight in weights] for epoch,weights in all_weights]
matrix=np.array(matrix)
plt.figure(0)
sns.heatmap(matrix.T[2:-2,:], cmap='viridis')
plt.savefig("plot3.png")
plt.figure(1)
sns.heatmap(matrix.T[-2:,:], cmap='viridis')
plt.savefig("plot2.png")
# for epoch,weights in all_weights:  

#     fig, axs = plt.subplots(1, len(weights[0]), figsize=(20, 4))  
#     fig.suptitle(f'Epoch {epoch+1}')  
#     for i, (name, layer_weights) in enumerate(weights[epoch].items()):  
#         if len(layer_weights.shape) > 1:  # 只绘制权重矩阵，不绘制偏置向量  
#             weight_norm = torch.norm(layer_weights, dim=-1).numpy()  
#             sns.heatmap(weight_norm, cmap='viridis', ax=axs[i])  
#             axs[i].set_title(f'{name}')  
# plt.show()