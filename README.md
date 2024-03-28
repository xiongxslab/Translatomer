# Translatomer

## Introduction
A transformer-based multi-modal deep learning framework that predicts ribosome profiling track using genomic sequence and cell-type-specific RNA-seq as input.

## Prerequisites
To run this project, you need the following prerequisites:
- Python 3.9
- PyTorch 1.13.1+cu117
- Other required Python libraries (please refer to requirements.txt)

You can install all the required packages using the following command:
```
conda create -n pytorch python=3.9.16
conda activate pytorch
```
```python
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
```python
pip install -r requirements.txt 
```

## Data Preparation
To generate training data, use the following command:
```
python generate_features.py [options]

[options]:
- --assembly  Genome reference for the data. Default = 'hg38'.
- --celltype  Name of the cell line. Default = 'K562'.
- --study  GEO accession number for the data. Default = 'GSE153597'.
- --region_len  The desired sequence length (region length). Default = 65536.
- --nBins  The number of bins for dividing the sequence. Default = 1024.

```

## Model Training
To train the Translatomer model, use the following command:
```
python train_all.py [options]

[options]:
- --seed  Random seed for training. Default value: 2077.
- --save_path  Path to the model checkpoint. Default = 'checkpoints'.
- --data-root  Root path of training data.  Default = 'data' (Required).
- --assembly  Genome assembly for training data. Default = 'hg38'.
- --model-type  Type of the model to use for training. Default = 'TransModel'.
- --patience  Epochs before early stopping. Default = 8.
- --max-epochs  Max epochs for training. Default = 128.
- --save-top-n  Top n models to save during training. Default = 20.
- --num-gpu  Number of GPUs to use for training. Default = 1.
- --batch-size  Batch size for data loading. Default = 8.
- --ddp-disabled  Flag to disable ddp (Distributed Data Parallel) for training. If provided, it will enable DDP with batch size adjustment.
- --num-workers  Number of dataloader workers. Default = 1.
```

## Load pretrained model
We provide [pretrained model](https://zjueducn-my.sharepoint.com/:u:/g/personal/xiongxs_zju_edu_cn/EQi7_h2XzLFDlM3lB_O2eTsBqg6sW1yQj4rm2FBhUcOLJA?e=aQAsHj) and [example data](https://zjueducn-my.sharepoint.com/:f:/g/personal/xiongxs_zju_edu_cn/EqgMcYc6CIVNs1fTMB00lHcB5K1AkEFDSnKsZU0F62kObQ?e=Lfppwl) 

## License
This project is licensed under MIT License.

## Contact
For any questions or inquiries, please contact xiongxs@zju.edu.cn.
