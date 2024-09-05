import os
import pathlib
import pandas as pd 
import argparse
import pyBigWig
import torch
import time
import math
import numpy as np
from kipoiseq import Interval
from einops import rearrange
import pyfaidx

def generate_inputs(assembly, name, study, region_file, fasta_file, bw_file, region_len=65536):
    regions = pd.read_csv(region_file, sep = '\t', names = ['chr', 'start', 'end', 'strand'])
    rnaseq_file = f'data/{assembly}/{name}/{study}/input_features/tmp/{name}_{region_len}_log_24chr_rnaseq_final.pt'
    if os.path.exists(rnaseq_file):
        return 
    else:
        print(f'Feature path: {bw_file} \n Normalization status: log', flush=True)

    t0 = time.time()
    bw = pyBigWig.open(bw_file)
    target = []
    for j, region in regions.iterrows():  
        chrom = region['chr']
        start = region['start']
        end = region['end']
        chromosome_length = bw.chroms(chrom)
        interval = Interval(chrom, start, end).resize(region_len)
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
    print(bw_file, time.time() - t0, flush=True)
    target = np.array(target).astype(np.float32)
    target = np.nan_to_num(target,0)
    target = np.log(target + 1)
    target.tofile(rnaseq_file) #updated
    bw.close()
    
def generate_outputs(assembly, name, study, region_file, fasta_file, bw_file, nBins=1024, region_len=65536):
    regions = pd.read_csv(region_file, sep = '\t', names = ['chr', 'start', 'end', 'strand'])
    riboseq_file = f'data/{assembly}/{name}/{study}/output_features/tmp/{name}_{region_len}_{nBins}_log_24chr_riboseq_final.pt'
    if os.path.exists(riboseq_file):
        return 
    else:
        print(f'Feature path: {bw_file} \n Normalization status: log', flush=True)

    t0 = time.time()
    bw = pyBigWig.open(bw_file)
    target = []
    for j, region in regions.iterrows():
        chrom = region['chr']
        start = region['start']
        end = region['end']
        chromosome_length = bw.chroms(chrom)
        interval = Interval(chrom, start, end).resize(region_len)
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        signals = np.array(bw.values(chrom, trimmed_interval.start, trimmed_interval.end)).astype(np.float32).tolist()
        pad_upstream = np.array([0] * max(-interval.start, 0)).astype(np.float32).tolist()
        pad_downstream = np.array([0] * max(interval.end - chromosome_length, 0)).astype(np.float32).tolist()
        tmp = pad_upstream + signals + pad_downstream
        
        arr = np.array(tmp).astype(np.float32)
        reshaped_arr = arr.reshape(-1, 64)
        averages = np.mean(reshaped_arr, axis=1)
        target.append(averages)
    print(bw_file, time.time() - t0, flush=True)
    target = np.array(target).astype(np.float32)
    target = np.nan_to_num(target,0)
    target = np.log(target + 1)
    torch.save(torch.Tensor(target), riboseq_file)
    bw.close() 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate features')
    parser.add_argument('--assembly', type=str, default='hg38', help='genome reference')
    parser.add_argument('--celltype', type=str, default='K562', help='name of the cell line')
    parser.add_argument('--study', type=str, default='GSE153597', help='GEO accession number')
    parser.add_argument('--region_len', type=int, default=65536, help='sequence length')
    parser.add_argument('--nBins', type=int, default=1024, help='number of bins')
    args = parser.parse_args()

    name = args.celltype
    study = args.study
    assembly = args.assembly

    print(name, flush=True)
    rna = f'data/{args.assembly}/{args.celltype}/{args.study}/input_features/rnaseq.bw'
    riboseq = f'data/{args.assembly}/{args.celltype}/{args.study}/output_features/riboseq.bw'

    if args.assembly == "hg38":
        region_file = f'data/{args.assembly}/gene_region_24chr.1.bed' #updated
        fasta_file = f'data/{args.assembly}/hg38.fa'
    elif args.assembly == "mm10":
        region_file = f'data/{args.assembly}/gene_region_21chr.1.bed'#updated
        fasta_file = f'data/{args.assembly}/mm10.fa'
    else:
        region_file = ""
        fasta_file = ""

    generate_inputs(assembly, name, study, region_file, fasta_file, rna, region_len=args.region_len)
    generate_outputs(assembly, name, study, region_file, fasta_file, riboseq, nBins=args.nBins, region_len=args.region_len)

#python generate_features_4rv.py --assembly hg38 --celltype K562 --study GSE153597 --region_len 65536 --nBins 1024
