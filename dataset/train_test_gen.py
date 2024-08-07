import numpy as np
import os
import tqdm
import random
import itertools  
import pickle
_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}
def get_index(factors):
  """ Converts factors to indices in range(num_data)
  Args:
    factors: np array shape [6,batch_size].
             factors[i]=factors[i,:] takes integer values in 
             range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

  Returns:
    indices: np array shape [batch_size].
  """
  indices = 0
  base = 1
  for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
    indices += factors[factor] * base
    base *= _NUM_VALUES_PER_FACTOR[name]
  return indices

factor_size = [10,10,10,8,4,15]
max_seq_len = 40


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_maps', type=int)
args=parser.parse_args()
num_maps=args.num_maps

if not os.path.exists("sample/train_test_imgs.pkl"):

  imgs = [list(img) for img in itertools.product(*[range(dim) for dim in factor_size])]  
  random.shuffle(imgs)
  train_imgs = np.array(imgs[:int(len(imgs)*0.8)])
  test_imgs = np.array(imgs[int(len(imgs)*0.8):])
  pickle.dump((train_imgs, test_imgs), open("sample/train_test_imgs.pkl", "wb"))
else:
   train_imgs,test_imgs=pickle.load(open("sample/train_test_imgs.pkl", "rb"))


tasks=[]
for i in range(num_maps):
    task={}
    for factor in range(len(factor_size)):
        task[factor]=np.random.permutation(factor_size[factor])
    tasks.append(task)

sample_item_num = int(1e5)
tasks_list = []
record_tasks = np.zeros((sample_item_num, 15))
record_labels = np.zeros((sample_item_num, max_seq_len))
record_indices = np.zeros((sample_item_num, max_seq_len))
record_trans = np.zeros(sample_item_num)
seq_len_array = np.random.randint(20, max_seq_len+1, size = (sample_item_num,))
factor_num = np.random.randint(0, len(factor_size), size = (sample_item_num,))
for i, (seq_len, factor) in tqdm.tqdm(enumerate(zip(seq_len_array, factor_num)),total=sample_item_num):
    seq_len=max_seq_len
    factors=train_imgs[np.random.choice(train_imgs.shape[0], size = (seq_len,), replace = False)]
    indices = get_index(factors.transpose(1,0))
    task=tasks[np.random.randint(0,num_maps)][factor]
    record_tasks[i, :factor_size[factor]] = task
    task_label = task[factors[:,factor]]
    record_indices[i, :factors.shape[0]] = indices
    record_labels[i, :factors.shape[0]] = task_label
np.savez(f"sample/train_{num_maps}.npz", indices = record_indices, labels = record_labels, factors = factor_num,record_trans=record_trans)

sample_item_num = int(4e4)
tasks_list = []
record_tasks = np.zeros((sample_item_num, 15))
record_labels = np.zeros((sample_item_num, max_seq_len))
record_indices = np.zeros((sample_item_num, max_seq_len))
record_trans = np.zeros(sample_item_num)
seq_len_array = np.random.randint(20, max_seq_len+1, size = (sample_item_num,))
factor_num = np.random.randint(0, len(factor_size), size = (sample_item_num,))
for i, (seq_len, factor) in tqdm.tqdm(enumerate(zip(seq_len_array, factor_num)),total=sample_item_num):
    seq_len=max_seq_len
    factors=test_imgs[np.random.choice(test_imgs.shape[0], size = (seq_len,), replace = False)]
    indices = get_index(factors.transpose(1,0))
    task=tasks[np.random.randint(0,num_maps)][factor]
    record_tasks[i, :factor_size[factor]] = task
    task_label = task[factors[:,factor]]
    record_indices[i, :factors.shape[0]] = indices
    record_labels[i, :factors.shape[0]] = task_label
np.savez(f"sample/test_{num_maps}.npz", indices = record_indices, labels = record_labels, factors = factor_num,record_trans=record_trans)