# Translatomer
This is our implementation for the paper:

Jialin He, Lei Xiong#, Shaohui Shi, Chengyu Li, Kexuan Chen, Qianchen Fang, Jiuhong Nan, Ke Ding, Jingyun Li, Yuanhui Mao, Carles A. Boix, Xinyang Hu, Manolis Kellis, Jingyun Li and Xushen Xiong#. [Deep learning modeling of ribosome profiling reveals regulatory underpinnings of translatome and interprets disease variants](https://www.biorxiv.org/content/10.1101/2024.02.26.582217v1).
(Preprint)

## Introduction
Translatomer is a transformer-based multi-modal deep learning framework that predicts ribosome profiling track using genomic sequence and cell-type-specific RNA-seq as input.
![Overview](https://github.com/xiongxslab/Translatomer/blob/9d26528ab055353b61e7602886099afba1c299ee/img/Model_overview.png)

## Citation
If you want to use our codes and datasets in your research, please cite:
```

```

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
Example data for model training can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.13751434)
- Put all input files in a **data** folder. The input files have to be organized as follows:
```
  + data
    + hg38
      + K562
        + GSE153597
          + input_features
            ++ rnaseq.bw 
          + output_features
            ++ riboseq.bw 
      + HepG2
        + GSE174419
          + input_features
            ++ rnaseq.bw 
          + output_features
            ++ riboseq.bw 
      *...
      ++ gencode.v43.annotation.gff3
      ++ hg38.fa
      ++ hg38.fai
      ++ mean.sorted.bw
    + mm10
      *...
```
- To generate training data, use the following command:
```
python generate_features_4rv.py [options]

[options]:
- --assembly  Genome reference for the data. Default = 'hg38'.
- --celltype  Name of the cell line. Default = 'K562'.
- --study  GEO accession number for the data. Default = 'GSE153597'.
- --region_len  The desired sequence length (region length). Default = 65536.
- --nBins  The number of bins for dividing the sequence. Default = 1024.

```

Example to run the codes:
```
find data/ -type d -name 'output_features' -exec mkdir -p '{}/tmp' \;
find data/ -type d -name 'input_features' -exec mkdir -p '{}/tmp' \;
nohup python generate_features_4rv.py --assembly hg38 --celltype HepG2 --study GSE174419 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype K562 --study GSE153597 --region_len 65536 --nBins 1024 &
```

## Model Training
To train the Translatomer model, use the following command:
```
python train_all_11fold.py [options]

[options]:
- --seed  Random seed for training. Default value: 2077.
- --save_path  Path to the model checkpoint. Default = 'checkpoints'.
- --data-root  Root path of training data.  Default = 'data' (Required).
- --assembly  Genome assembly for training data. Default = 'hg38'.
- --model-type  Type of the model to use for training. Default = 'TransModel'.
- --fold  Which fold of the model training. Default='0',
- --patience  Epochs before early stopping. Default = 8.
- --max-epochs  Max epochs for training. Default = 128.
- --save-top-n  Top n models to save during training. Default = 20.
- --num-gpu  Number of GPUs to use for training. Default = 1.
- --batch-size  Batch size for data loading. Default = 32.
- --ddp-disabled  Flag to disable ddp (Distributed Data Parallel) for training. If provided, it will enable DDP with batch size adjustment.
- --num-workers  Number of dataloader workers. Default = 1.
```
Example to run the codes:
```
nohup python train_all_11fold.py --save_path results/bigmodel_h512_l12_lr1e-5_wd0.05_ws2k_p32_fold0 --data-root data --assembly hg38 --dataset data_roots_mini.txt --model-type TransModel --fold 0 --patience 6 --max-epochs 128 --save-top-n 128 --num-gpu 1 --batch-size 32 --num-workers 1 >DNA_logs/bigmodel_h512_l12_lr1e-5_wd0.05_ws2k_p32_fold0.log 2>&1 &
nohup python train_all_11fold.py --save_path results/bigmodel_h512_l12_lr1e-5_wd0.05_ws2k_p32_fold1 --data-root data --assembly hg38 --dataset data_roots_mini.txt --model-type TransModel --fold 1 --patience 6 --max-epochs 128 --save-top-n 128 --num-gpu 1 --batch-size 32 --num-workers 1 >DNA_logs/bigmodel_h512_l12_lr1e-5_wd0.05_ws2k_p32_fold1.log 2>&1 &
```

## Tutorial
- Load pretrained model
Pretrained model can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.13751434)
- An example notebook containing code for applying Translatomer is [here](https://github.com/xiongxslab/Translatomer/blob/main/Tutorial.ipynb).

## License
This project is licensed under MIT License.

## Contact
For any questions or inquiries, please contact xiongxs@zju.edu.cn.
