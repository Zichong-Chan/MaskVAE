# MaskVAE

MaskVAE training on CelebAMask-HQ.


## Data preparing

[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) is used for training. 

We follow the preprocessing steps of [CelebAMask-HQ/face_parsing](https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing).
After running the preprocessing scripts, **6 folders and 3 text files** (`train_img`, `train_label`, `test_img`, `test_label`, `val_img`, `val_label`, `train_list.txt`, `test_list`, `val_list.txt`) will be created.

MaskVAE only needs labels for training (`train_label`).

## Training
Run the following command to start training.
```shell
python train.py --dataset=[LABEL_DATA_PATH] --total_step=40000 --batch_size=8 --version=MaskVAE_v0
```

For example, 
```shell
python train.py --dataset=./data/train_label 
```
Here, we place the training data (folder) to `./data` by default, and training using the default configurations.

See the source code for more configuration details.


