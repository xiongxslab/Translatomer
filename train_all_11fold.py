import sys
import torch
import argparse
import numpy as np
import os 
import json
import pandas as pd
import pyfaidx
import kipoiseq
from kipoiseq import Interval
import pyBigWig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torchmetrics import Metric

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
import pytorch_lightning.callbacks as callbacks

from typing import Optional
import model.models as models

from tensor_loader import TensorLoader 

import time
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEQ_LEN = 65536
TARGET_LEN = 1024

class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = fasta_file #pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in pyfaidx.Fasta(self.fasta).items()}
        self.seq_len = SEQ_LEN

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(pyfaidx.Fasta(self.fasta).get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.end).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream
    
    def close(self):
        return pyfaidx.Fasta(self.fasta).close() #self.fasta.close()

# Correlation computation along positions from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/metrics.py
class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable: Optional[bool] = False
    full_state_update:bool = False
    higher_is_better: Optional[bool] = True
    def __init__(self, n_channels:int, dist_sync_on_step=False):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        super().__init__(dist_sync_on_step=dist_sync_on_step, full_state_update=False)
        self.reduce_dims=(0, 1)
        self.add_state("product", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("count", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = (self.product
                    - true_mean * self.pred
                    - pred_mean * self.true
                    + self.count * true_mean * pred_mean)

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, regions, input_features, output_features, fasta_reader, seq_len=65536, target_length=1024, use_aug = True):
        self.target_length = target_length
        self.seq_len = seq_len
        
        self.fasta_reader = fasta_reader #FastaStringExtractor(fasta_path)
        self.regions = regions
        self.input_features = input_features
        self.output_features = output_features
        
        self.use_aug = use_aug
        
    @staticmethod
    def one_hot_encode(sequence):
        en_dict = {'A' : 0, 'T' : 1, 'C' : 2, 'G' : 3, 'N' : 4}
        en_seq = [en_dict[ch] for ch in sequence]
        np_seq = np.array(en_seq, dtype = int)
        seq_emb = np.zeros((len(np_seq), 5))
        seq_emb[np.arange(len(np_seq)), np_seq] = 1
        return seq_emb.astype(np.float32)

    def __len__(self):
        return len(self.regions)
    
    def reverse(self, seq, input_features, output_features, strand):
        '''
        Reverse sequence and matrix
        '''
        if strand == '-':
            seq_r = np.flip(seq, 0).copy() # n x 5 shape
            input_features_r = torch.flip(input_features, dims=[0]) 
            output_features_r = torch.flip(output_features, dims=[0]) # n
            # Complementary sequence
            seq_r = self.complement(seq_r)
        else:
            seq_r = seq
            input_features_r = input_features
            output_features_r = output_features 
        return seq_r, input_features_r, output_features_r

    def complement(self, seq):
        '''
        Complimentary sequence
        '''
        seq_comp = np.concatenate([seq[:, 1:2],
                                   seq[:, 0:1],
                                   seq[:, 3:4],
                                   seq[:, 2:3],
                                   seq[:, 4:5]], axis = 1)
        return seq_comp

    def __getitem__(self, idx):
        loc_row = self.regions.iloc[idx]
        target_interval = Interval(loc_row['chr'], loc_row['start'], loc_row['end']).resize(self.seq_len)
        sequence = self.fasta_reader.extract(target_interval)
        sequence_one_hot = self.one_hot_encode(sequence)
        input_features = self.input_features[idx]
        output_features = self.output_features[idx]
        chrom = self.regions.iloc[idx]['chr']
        start = loc_row['start']
        end = loc_row['end']
        strand = loc_row['strand']
        if self.use_aug:
            sequence_one_hot, input_features, output_features = self.reverse(sequence_one_hot, input_features, output_features, strand)
            
        return {
            'sequence': sequence_one_hot,
            'input_features': input_features,
            'output_features': output_features,
            'chrom': chrom,
            'start': start,
            'end': end
        }

class DataModule(LightningDataModule):
    def __init__(
        self,
        region_file: str=None,
        input_file: list=[],
        output_file: list=[],
        fasta_path: str=None,
        data_roots: list=[],
        seq_len: int=65536,
        target_length: int=1024,
        train_chrlist: list=[],
        val_chrlist: list=[],
        test_chrlist: list=[],
        batch_size: int=32,
        eval_batch_size: int=None,
        num_workers: int=4,
        pin_memory: bool=False,
        **kwargs
    ):
        super().__init__()
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.regions = pd.read_csv(region_file, sep='\t', names = ['chr', 'start', 'end', 'strand'])
        self.input_features = input_file
        self.output_features = output_file
        
        self.data_roots = data_roots
        self.fasta_reader = FastaStringExtractor(fasta_path)
        
        train_idx = [ i for i, c in enumerate(self.regions['chr']) if c in train_chrlist ]
        val_idx = [ i for i, c in enumerate(self.regions['chr']) if c in val_chrlist ]
        test_idx = [ i for i, c in enumerate(self.regions['chr']) if c in test_chrlist ]

        train_dataset_list = []
        val_dataset_list = []
        test_dataset_list = []
        for i, data_root in enumerate(data_roots): 
            train_dataset = Dataset(self.regions.iloc[train_idx], self.input_features[i][train_idx], self.output_features[i][train_idx], self.fasta_reader, seq_len, target_length)
            val_dataset = Dataset(self.regions.iloc[val_idx], self.input_features[i][val_idx], self.output_features[i][val_idx], self.fasta_reader, seq_len, target_length)
            test_dataset = Dataset(self.regions.iloc[test_idx], self.input_features[i][test_idx], self.output_features[i][test_idx], self.fasta_reader, seq_len, target_length)
            
            train_dataset_list.append(train_dataset)
            val_dataset_list.append(val_dataset)
            test_dataset_list.append(test_dataset)
            
        self.train_dataset = ConcatDataset(train_dataset_list)
        self.val_dataset = ConcatDataset(val_dataset_list)
        self.test_dataset = ConcatDataset(test_dataset_list)       
            
    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.eval_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=1
        )
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=1
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=1
        )
        return loader
    
class TrainModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.get_model(args)
        self.args = args
        self.criterion = nn.MSELoss()
        self.pcc = MeanPearsonCorrCoefPerChannel(1)

    def forward(self, x):
        pred = self.model(x)
        return pred 
    
    def proc_batch(self, batch):
        seq = batch['sequence']
        epi = batch['input_features'].unsqueeze(2)
        targets = batch['output_features']
        inputs = torch.cat([seq, epi], dim = 2)
        targets = targets.float() 
        return inputs, targets

    def training_step(self, batch, batch_idx):
        inputs, targets = self.proc_batch(batch)
        pred = self(inputs)   
        loss = self.criterion(pred, targets).mean()
        pcc = self.pcc(pred, targets).mean()
        ranks1 = pred.argsort().argsort().type(torch.float32)
        ranks2 = targets.argsort().argsort().type(torch.float32)
        scc = self.pcc(ranks1, ranks2).mean()
        metrics = {
            'loss/train_step' : loss, 
            'pearson/train_step': pcc,
            'spearman/train_step': scc
        }
        self.log_dict(metrics, batch_size = inputs.shape[0], prog_bar=True, sync_dist=True)
        return {'loss':loss,'pcc':pcc,'scc':scc }

    def validation_step(self, batch, batch_idx):
        inputs, targets = self.proc_batch(batch)
        pred = self(inputs)   
        loss = self.criterion(pred, targets).mean()
        pcc = self.pcc(pred, targets).mean()
        ranks1 = pred.argsort().argsort().type(torch.float32)
        ranks2 = targets.argsort().argsort().type(torch.float32)
        scc = self.pcc(ranks1, ranks2).mean()
        return {'loss':loss,'pcc':pcc,'scc':scc  }
    
    def test_step(self, batch, batch_idx):
        inputs, targets = self.proc_batch(batch)
        pred = self(inputs)   
        loss = self.criterion(pred, targets).mean()
        pcc = self.pcc(pred, targets).mean()
        ranks1 = pred.argsort().argsort().type(torch.float32)
        ranks2 = targets.argsort().argsort().type(torch.float32)
        scc = self.pcc(ranks1, ranks2).mean()
        return {'loss':loss,'pcc':pcc,'scc':scc  }
    
    # Collect epoch statistics
    def training_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'train_loss': ret_metrics['loss'], 'train_pcc': ret_metrics['pcc'], 'train_scc': ret_metrics['scc']
                  }
        self.log_dict(metrics, prog_bar=True, on_epoch=True,on_step=False, sync_dist=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'val_loss' : ret_metrics['loss'], 'val_pcc': ret_metrics['pcc'], 'val_scc': ret_metrics['scc']
                  }
        self.log_dict(metrics, prog_bar=True, on_epoch=True,on_step=False, sync_dist=True)
        
    def test_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'test_loss' : ret_metrics['loss'], 'test_pcc': ret_metrics['pcc'], 'test_scc': ret_metrics['scc']
                  }
        self.log_dict(metrics, prog_bar=True, on_epoch=True,on_step=False, sync_dist=True)  
        
    def _shared_epoch_end(self, step_outputs):
        pcc = torch.tensor([out['pcc'] for out in step_outputs])
        loss = torch.tensor([out['loss']for out in step_outputs])
        scc = torch.tensor([out['scc'] for out in step_outputs])
        pcc = pcc[~torch.isnan(pcc)]
        scc = scc[~torch.isnan(scc)]
        avg_pcc = pcc.mean()
        avg_loss = loss.mean()
        avg_scc = scc.mean()
        return {'loss' : avg_loss, 'pcc' : avg_pcc, 'scc':avg_scc}
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = 1e-5, weight_decay = 0.05) #final
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps= 2000, #self.hparams.warmup_steps
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'get_cosine_schedule_with_warmup',
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}
    
    def get_model(self, args):
        model_name = args.model_type
        num_genomic_features = 1
        ModelClass = getattr(models, model_name)
        model = ModelClass(num_genomic_features, mid_hidden = 512)
        return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Translatomer')
    
    parser.add_argument('--seed', dest='run_seed', default=2077,
                        type=int,
                        help='Random seed for training')
    parser.add_argument('--save_path', dest='run_save_path', default='checkpoints',
                        help='Path to the model checkpoint')   
    parser.add_argument('--data-root', dest='dataset_data_root', default='data',
                        help='Root path of training data', required=True)
    parser.add_argument('--assembly', dest='dataset_assembly', default='hg38',
                        help='Genome assembly for training data')
    parser.add_argument('--dataset', dest='dataset', default='data_roots.txt',
                        help='Muilti input for data training')
    parser.add_argument('--model-type', dest='model_type', default='TransModel',
                        help='Transformer')
    parser.add_argument('--fold', dest='n_fold', default='0',
                        help='Which fold of the model training')
    
  # Training Parameters
    parser.add_argument('--patience', dest='trainer_patience', default=8,
                        type=int,
                        help='Epoches before early stopping')
    parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=128,
                        type=int,
                        help='Max epochs')
    parser.add_argument('--save-top-n', dest='trainer_save_top_n', default=20,
                        type=int,
                        help='Top n models to save')
    parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=1,
                        type=int,
                        help='Number of GPUs to use')
  # Dataloader Parameters
    parser.add_argument('--batch-size', dest='dataloader_batch_size', default=8, 
                        type=int,
                        help='Batch size')
    parser.add_argument('--ddp-disabled', dest='dataloader_ddp_disabled',
                        action='store_false',
                        help='Using ddp, adjust batch size')
    parser.add_argument('--num-workers', dest='dataloader_num_workers', default=1,
                        type=int,
                        help='Dataloader workers')
    parser.add_argument('--checkpoint', type=str, default=None)
   
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    seq_len = SEQ_LEN
    target_length = TARGET_LEN
    region_file = f'data/{args.dataset_assembly}/gene_region_24chr.1.bed'#human-newBed
    fasta_path = f'data/{args.dataset_assembly}/{args.dataset_assembly}.fa'
    
    with open(region_file, 'r') as file:
        region_count = sum(1 for line in file if line.strip())
    tensor_loader = TensorLoader(region_count, seq_len)
    
    input_data = []
    output_data = []
    names = locals()
    celltype_list = []
    study_list = []
    with open({args.dataset}, 'r') as file: #human
        data_roots = [line.strip('\n') for line in file]
    for i, data_root in enumerate(data_roots):
        celltype, study = data_root.split('\t')
        celltype_list.append(celltype)
        study_list.append(study)
        output_data.append(torch.load(f'data/{args.dataset_assembly}/{celltype}/{study}/output_features/tmp/{celltype}_{seq_len}_{target_length}_log_24chr_riboseq_final.pt')) #human-newBed
        input_data.append(tensor_loader.load(f'data/{args.dataset_assembly}/{celltype}/{study}/input_features/tmp/{celltype}_{seq_len}_log_24chr_rnaseq_final.pt')) #human -newBed   
    pl.seed_everything(args.run_seed, workers=True)
    chrlist =['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22']
    random.shuffle(chrlist)
    n_fold = int(args.n_fold)
    print(f'FOLD {n_fold}')
    print('--------------------------------')
    val_chrlist = chrlist[2*n_fold:2*n_fold+2]
    test_chrlist = chrlist[2*n_fold+2:2*n_fold+4]
    while test_chrlist == []:
        test_chrlist = chrlist[0:2]
    train_chrlist = list(set(chrlist)-set(val_chrlist)-set(test_chrlist))
    print("all_chrlist:",chrlist)
    print("train_chrlist:",train_chrlist)
    print("val_chrlist:",val_chrlist)
    print("test_chrlist:",test_chrlist)
    
    dataset = DataModule(
        region_file = region_file,
        fasta_path = fasta_path,
        input_file = input_data, 
        output_file = output_data,
        data_roots = data_roots,
        seq_len = seq_len,
        target_length = target_length,
        train_chrlist = train_chrlist,
        val_chrlist = val_chrlist,
        test_chrlist = test_chrlist,
        batch_size = args.dataloader_batch_size,
        num_workers= args.dataloader_num_workers,
    )
    # loading data
    if args.checkpoint:
        model = TrainModule()
        trainer = Trainer(resume_from_checkpoint=args.checkpoint)
        trainer.fit()

    else:
        # Early_stopping
        early_stop_callback = callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0.00, 
                                            patience=args.trainer_patience,
                                            verbose=False,
                                            mode="min")
        # Checkpoints
        checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run_save_path}/models',
                                            save_top_k=args.trainer_save_top_n, 
                                            monitor='val_loss',
                                            mode = 'min')
        # LR monitor
        lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')
        csv_logger = pl.loggers.CSVLogger(save_dir = f'{args.run_save_path}/csv')
        all_loggers = csv_logger
        
        pl.seed_everything(args.run_seed, workers=True)
        pl_module = TrainModule(args)
        trainer = Trainer.from_argparse_args(
            args, 
            accelerator = 'gpu',
            strategy='ddp',
            devices = args.trainer_num_gpu,
            callbacks = [early_stop_callback,
                         checkpoint_callback,
                         lr_monitor],
            max_epochs = args.trainer_max_epochs,
            gradient_clip_val=0.8, ##revision_nan
            num_sanity_val_steps = 0,
            precision=32,#precision=16,
            logger = all_loggers
        )
        trainer.fit(pl_module, dataset.train_dataloader(), dataset.val_dataloader())
        trainer.test(pl_module, dataset.test_dataloader())
        
    tensor_loader.close()
    
#nohup python train_all_11fold.py --save_path results/bigmodel/test_bigmodel_h512_l12_lr1e-5_wd0.05_ws2k_p32_fold0 --data-root data --assembly hg38 --dataset data_roots.txt --model-type TransModel --fold 0 --patience 6 --max-epochs 128 --save-top-n 128 --num-gpu 1 --batch-size 32 --num-workers 1 >DNA_logs/test_bigmodel_h512_l12_lr1e-5_wd0.05_ws2k_p32_fold0.log 2>&1 &
