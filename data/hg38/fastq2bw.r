#! /bin/bash
#conda activate deeptools	
#riboseq:
#SE
#1. derRNA
cd riboseq/
mkdir derRNA
index=/data/slurm/hejl/ref/hg38/Homo_sapiens.rRNA

ls *fastq.gz |while read id
do
  sam=${id%%.*};
  nohup bowtie2 -x $index --un-gz derRNA/${sam}.derRNA.fq.gz -U $id -p 20 -S derRNA/${sam}.rRNA.mapped.sam >derRNA/${sam}.log 2>&1 &
done

#2. detRNA
mkdir detRNA
index=/data/slurm/hejl/ref/hg38/Homo_sapiens.tRNA

cd derRNA/
ls *fq.gz |while read id
do
  sam=${id%%.*}
  nohup bowtie2 -x $index --un-gz ../detRNA/${sam}.detRNA.fq.gz -U $id -p 20 -S ../detRNA/${sam}.tRNA.mapped.sam >../detRNA/${sam}.log 2>&1 &
done

# 3. map
cd ..
mkdir map
index=/data/slurm/hejl/ref/hg38/hg38

cd detRNA/
ls *fq.gz |while read id
do
  sam=${id%%.*}
  nohup bowtie2 --local -x $index -U $id -p 20 -S ../map/${sam}.mapped.sam >../map/${sam}.log 2>&1 &
done

#4. sam2bam
cd ../map
ls *.sam | while read id
do
  sam=${id%%.*}
  samtools view -bS ${id} -t 16 > ./${sam}.bam && \
  samtools sort -@ 16 -O bam ./${sam}.bam -o ./${sam}.sorted.bam  && \ 
  samtools index ./${sam}.sorted.bam ./${sam}.sorted.bam.bai 
done

#5. merge replicates
mkdir merge
samtools merge merge/merged.bam *sorted.bam &&
samtools index merge/merged.bam merge/merged.bam.bai 

#6. bam2bw
cd ..
mkdir bigwig
cd map/merge/
#Add normalization
nohup bamCoverage -p 16 -b merged.bam --normalizeUsing RPKM --binSize 1 -o ../../bigwig/riboseq.normalized.bw >../../ribo.bam2bw.normalized.log 2>&1 &

	

