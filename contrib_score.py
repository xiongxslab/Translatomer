import kipoiseq
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from train import TrainModule, FastaStringExtractor

import matplotlib.pyplot as plt
import seaborn as sns

import pyBigWig
from kipoiseq import Interval
import math
import random
from scipy.stats import pearsonr

random.seed(2077)
SEQUENCE_LENGTH = 65536

device = 'cuda:0'
checkpoint = '/data/slurm/hejl/riboseq/results_DNA/bigmodel/bigmodel_h512_l12_lr1e-5/models/epoch=38-step=746889.ckpt'
model = TrainModule.load_from_checkpoint(checkpoint).to(device)
model = model.eval()

class model_x(nn.Module):
    def __init__(self, path, device):
        super().__init__()
        self.model = TrainModule.load_from_checkpoint(checkpoint).to(device).eval()
    def forward(self, x):
        reshaped_arr = torch.Tensor(x[:, :, 5]).reshape(-1, 64)
        y = torch.mean(reshaped_arr, axis=1) + 1e-6
        return self.model(x)/y
    
model_x = model_x(checkpoint, device)

def one_hot_encode(sequence)
    en_dict = {'A' : 0, 'T' : 1, 'C' : 2, 'G' : 3, 'N' : 4} 
    en_seq = [en_dict[ch] for ch in sequence]
    np_seq = np.array(en_seq, dtype = int)
    seq_emb = np.zeros((len(np_seq), 5))
    seq_emb[np.arange(len(np_seq)), np_seq] = 1
    return seq_emb.astype(np.float32)

def one_hot_encode_4dim(sequence):
    en_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]} #TFmodisco
    encoded_sequence = [en_dict[ch] for ch in sequence]
    return np.array(encoded_sequence, dtype=np.float32)

def scaled_contribution_input_grad(model, model_input, target_mask):
    model_input = model_input.clone().detach().requires_grad_(True)  
    target_mask_mass = torch.sum(target_mask)
    # Forward pass
    prediction = torch.sum(target_mask * model(model_input)) / target_mask_mass
    # Backward pass
    prediction.backward()
    input_grad = model_input.grad * model_input
    input_grad = input_grad[:, :, :5]
    input_grad = input_grad.squeeze(0)
    return input_grad.sum(axis=-1).detach()

def scaled_hypothetical_score(model, model_input, target_mask):
    model_input = model_input.clone().detach().requires_grad_(True)  
    target_mask_mass = torch.sum(target_mask)
    # Forward pass
    prediction = torch.sum(target_mask * model(model_input)) / target_mask_mass
    # Backward pass
    prediction.backward()
    input_grad = model_input.grad
    input_grad = input_grad[:, :, :4]
    input_grad = input_grad.squeeze(0)
    input_grad = input_grad[:, [0, 2, 3, 1]]
    return input_grad.detach()

fasta_file = '/data/slurm/hejl/riboseq/data/hg38/hg38.fa'
fasta_extractor = FastaStringExtractor(fasta_file)
gff_file = '/data/slurm/hejl/riboseq/gencode.v43.annotation.gff3'
rna_bw_file = '/data/slurm/hejl/riboseq/Translatomer/data/hg38/mean.sorted.bw' #32 cell type
geneid_file = '/data/slurm/hejl/riboseq/Translatomer/motif/fwd.all_gene.txt'  #allgene
            
def interval_generator(geneid_file):
    with open(geneid_file) as f:
        for line in f:
            id, chrom, start, end, score = line.split("\t")[:5] 
            start=int(start)-1   #TSS start
            end= start + SEQUENCE_LENGTH 
            yield kipoiseq.Interval(chrom=chrom, start=start, end=end)
        
def get_interval(gff_df, gene_name):
    # filter by gene_id
    gff_df = gff_df[gff_df[8].str.split(';').str[0].str.split('=').str[1] == gene_name]

    # generate dict
    gene_dict = {row[8].split(';')[0].split('=')[1]: {'chrom': row[0],'start': row[3], 'end': row[4], 'name':row[8].split(';')[3].split('=')[1], 
                                                      'cds_intervals': []} for _, row in gff_df.iterrows() if row[2] == 'gene'}
    cds_df = gff_df[gff_df[2] == 'CDS']
    for _, row in cds_df.iterrows():
        gene_name = row[8].split(';')[3].split('=')[1]
        gene_dict[gene_name]['cds_intervals'].append((row[3], row[4]))
    target_gene = next((gene_info for gene_info in gene_dict.values()
                      ), None)
    if target_gene:
        cds_intervals = target_gene['cds_intervals']
        return {
            'chrom': target_gene['chrom'],
            'gene_start': target_gene['start'],
            'gene_end': target_gene['end'],
            'cds_intervals': cds_intervals,
            'gene_name': target_gene['name']
        }
    
def generate_inputs(interval, fasta_file, bw_file, region_len=SEQUENCE_LENGTH):
    bw = pyBigWig.open(bw_file)
    target = []
    chrom = interval.chrom
    start = interval.start
    end = interval.end
    chromosome_length = bw.chroms(chrom)
    trimmed_interval = Interval(interval.chrom,
                                max(interval.start, 0),
                                min(interval.end, chromosome_length),
                                )
    signals = np.array(bw.values(chrom, trimmed_interval.start, trimmed_interval.end)).astype(np.float32).tolist()
    pad_upstream = np.array([0] * max(-interval.start, 0)).astype(np.float32).tolist()
    pad_downstream = np.array([0] * max(interval.end - chromosome_length, 0)).astype(np.float32).tolist()
    tmp = pad_upstream + signals + pad_downstream
    arr = np.array(tmp).astype(np.float32)
    target.append(arr)

    target = np.array(target).astype(np.float32)
    target = np.nan_to_num(target,0)
    target = np.log(target + 1)
    bw.close()
    return target   

contribution_scores_list = []
model_input_list = []
TFmodisco_input_list = []
hypothetical_scores_list = []
for interval in interval_generator(geneid_file):
    if interval is None:
        continue
    target_interval = interval.resize(SEQUENCE_LENGTH)
    ref_seq = fasta_extractor.extract(target_interval)
    
    ref_emb = torch.Tensor(one_hot_encode(ref_seq)).to(device)
    ref_emb_4dim = torch.Tensor(one_hot_encode_4dim(ref_seq)).to(device)
    epi = torch.Tensor(generate_inputs(target_interval, fasta_file, rna_bw_file)[0]).unsqueeze(1).to(device)
    model_input = torch.cat([ref_emb, epi], dim = 1).unsqueeze(0)
    
    TFmodisco_input = ref_emb_4dim
    target_mask = torch.ones_like(model(model_input)).to(device)
    contribution_scores_np = scaled_contribution_input_grad(model_x, model_input, torch.Tensor(target_mask)).unsqueeze(0).cpu().numpy()
    # Append the contribution_scores to the list
    TFmodisco_input_np = TFmodisco_input.transpose(0, 1).detach().cpu().numpy()
    TFmodisco_input_list.append(TFmodisco_input_np)
    
    contribution_scores_emb = np.repeat(contribution_scores_np, 4, axis=0)
    contribution_scores_emb = contribution_scores_emb * TFmodisco_input_np
    contribution_scores_list.append(contribution_scores_emb)
    
    scaled_hypothetical_scores = scaled_hypothetical_score(model_x, model_input, torch.Tensor(target_mask)).transpose(0, 1).detach().cpu().numpy()
    hypothetical_scores_list.append(scaled_hypothetical_scores)
    
# Convert the list to a NumPy array
contribution_scores_array = np.array(contribution_scores_list)
TFmodisco_input_array = np.array(TFmodisco_input_list)
hypothetical_scores_array = np.array(hypothetical_scores_list)

contribution_scores_array[np.isnan(contribution_scores_array) | np.isinf(contribution_scores_array)] = 0
TFmodisco_input_array[np.isnan(TFmodisco_input_array) | np.isinf(TFmodisco_input_array)] = 0
hypothetical_scores_array[np.isnan(hypothetical_scores_array) | np.isinf(hypothetical_scores_array)] = 0

# Save the array to a .npz file
np.savez('/data/slurm/hejl/riboseq/Translatomer/contib_score/TFmodisco_seq.all.fwd.65k.npz', arr_0 = TFmodisco_input_array)
np.savez('/data/slurm/hejl/riboseq/Translatomer/contib_score/TFmodisco_score.all.fwd.65k.npz', arr_0 = contribution_scores_array)
np.savez('/data/slurm/hejl/riboseq/Translatomer/contib_score/TFmodisco_hypscore.all.fwd.65k.npz', arr_0 = hypothetical_scores_array)
