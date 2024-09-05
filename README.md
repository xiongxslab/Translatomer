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
To generate training data, use the following command:
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
nohup python generate_features_4rv.py --assembly hg38 --celltype A549 --study GSE82232 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype BJ --study GSE69906 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype brain --study GSE51424 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype CN34 --study GSE77292 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype Cybrid --study GSE48933 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype H1933 --study GSE96716 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype HAP-1 --study GSE97140 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype HEK293T --study GSE52809 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype HeLa --study GSE83493 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype HepG2 --study GSE174419 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype hESC --study GSE78959_2 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype Huh7 --study GSE69602 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype K562 --study GSE153597 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype kidney --study GSE59821 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype macrophages --study GSE66809 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype MCF7 --study GSE96643 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype MCF10A --study GSE59817 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype MDA --study GSE77315 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype MM1.S --study GSE69047 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype neuron --study GSE90469_2 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype NPC --study GSE100007_1 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype PC3 --study GSE35469 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype PC9 --study GSE96716 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype T-ALL --study GSE56887 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype U2OS --study GSE66929 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype WTC_11 --study GSE131650 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype SW480 --study GSE196982 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype H9 --study GSE162050 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype SH-SY5Y --study GSE155727 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype MB135iDUX4 --study GSE178761 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype 12T --study GSE142822 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype erythroid --study GSE131809 --region_len 65536 --nBins 1024 &
nohup python generate_features_4rv.py --assembly hg38 --celltype liver --study GSE112705 --region_len 65536 --nBins 1024 &

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
- --batch-size  Batch size for data loading. Default = 6.
- --ddp-disabled  Flag to disable ddp (Distributed Data Parallel) for training. If provided, it will enable DDP with batch size adjustment.
- --num-workers  Number of dataloader workers. Default = 1.
```
Example to run the codes:
```
nohup python train_all_11fold.py --save_path results/bigmodel_h512_l12_lr1e-5_wd0.05_ws2k_p32_fold0 --data-root data --assembly hg38 --model-type TransModel --fold 0 --patience 6 --max-epochs 128 --save-top-n 128 --num-gpu 1 --batch-size 32 --num-workers 1 >DNA_logs/bigmodel_h512_l12_lr1e-5_wd0.05_ws2k_p32_fold0.log 2>&1 &
nohup python train_all_11fold.py --save_path results/bigmodel_h512_l12_lr1e-5_wd0.05_ws2k_p32_fold1 --data-root data --assembly hg38 --model-type TransModel --fold 1 --patience 6 --max-epochs 128 --save-top-n 128 --num-gpu 1 --batch-size 32 --num-workers 1 >DNA_logs/bigmodel_h512_l12_lr1e-5_wd0.05_ws2k_p32_fold1.log 2>&1 &
```

## Load pretrained model
We provide [pretrained model](https://zjueducn-my.sharepoint.com/:u:/g/personal/xiongxs_zju_edu_cn/EQi7_h2XzLFDlM3lB_O2eTsBqg6sW1yQj4rm2FBhUcOLJA?e=aQAsHj) and [example data](https://zjueducn-my.sharepoint.com/:f:/g/personal/xiongxs_zju_edu_cn/EqgMcYc6CIVNs1fTMB00lHcB5K1AkEFDSnKsZU0F62kObQ?e=Lfppwl) .

Example to run the codes:
```
python generate_features.py --assembly hg38 --celltype A549 --study GSE82232 --region_len 65536 --nBins 1024
```
```
python train.py --save_path results/ --data-root data --assembly hg38 --celltype A549 --study GSE82232 --model-type TransModel --patience 8 --max-epochs 128 --save-top-n 128 --num-gpu 1 --batch-size 32 --num-workers 1
```

## License
This project is licensed under MIT License.

## Contact
For any questions or inquiries, please contact xiongxs@zju.edu.cn.
