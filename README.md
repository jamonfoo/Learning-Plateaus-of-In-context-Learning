# Code for the paper "Breaking through the learning plateaus of in-context learning in Transformer" by Jingwen Fu et al.


## Citations

If you use this code or the provided environments in your research, please cite the following paper:

    @inproceedings{fubreaking,
    title={Breaking through the learning plateaus of in-context learning in Transformer},
    author={Fu, Jingwen and Yang, Tao and Wang, Yuwang and Lu, Yan and Zheng, Nanning},
    booktitle={Forty-first International Conference on Machine Learning}
    }

## Description

This code reproduces Figures 3, 4, and 5C from the paper.

## Datasets

We use the Shape3D dataset to construct the in-context learning task. A pretrained VAE encodes the images into embeddings, which are used as input for the Transformer. You can download the embeddings from the link:

https://1drv.ms/u/c/aac24bf3205d2006/EZGe_9oWDv9Lp-PNuSP4G1oBbEgHCVclKVaN1prMT7EfNg?e=2Q6Gmd.

After downloading the embeddings, place them in the 'dataset' directory.

## Running the code

### Reproduce the results of Figure 3,4



1. Generate the corresponding in-context learning task using the following command:

        python dataset/train_test_gen.py --num_maps 20000

   The --num_maps argument controls the number of possible mappings between the factor values and labels (denoted as $m$ in the paper). Setting --num_maps to 1 results in the $D_{fix}$ setting described in the paper.

2. Train the model with the following command:

        python train.py --config config/standard.yaml --num_maps 20000

3. After running the code, you will obtain checkpoints of the model during the learning process. To test the accuracy and the weights component score, use the following commands:
    
        python eval_incontext.py --config config/standard.yaml --num_maps 20000
        python eval_weights.py --config config/standard.yaml --num_maps 20000

### Reproduce the results of Figure 5C

1. Follow the previous steps (Step 1) to generate the data.
2. Use the following command to train the Transformer:

        python train_extra.py --config config/standard.yaml --mode "weights" --num_maps 20000
    The --mode argument controls whether to use weights extra loss or context extra loss, as defined in Section 5.3 of the paper.
3. Use the following command to test the accuracy:
        
        python eval_extra.py --config config/standard.yaml --mode "weights" --num_maps 20000



