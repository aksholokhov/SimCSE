## (Reproducibility study) SimCSE: Simple Contrastive Learning of Sentence Embeddings

In this repo, we reproduce the results for unsupervised SimCSE. The original authors of the paper are Tianyu Gao, Xingcheng Yao, and Danqi Chen. The authors of this reproducibility study are [@aksholokhov](https://github.com/aksholokhov) and [@birajpandey](https://github.com/birajpandey).


Link to original repo: https://github.com/princeton-nlp/SimCSE 

Link to original paper: https://arxiv.org/abs/2104.08821


## Quick Links

  - [Compute Details](#compute-details)
  - [Dependencies](#dependencies)
  - [Main Results](#main-results)
  - [Data download](#data-download)


## Compute Details

For this study, we use 4 NVIDIA Tesla A40 GPUs with 48 GB memory each. 

## Dependencies

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). Please use the correct `1.7.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.7.1` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.1
```


Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

## Data Download
To download the corresponding dataset, run the shell script `data/downloadwiki.sh`


```bash
sh data/downloadwiki.sh
```

## Data preprocessing


## Main Results
We show the performance of unsupervied SimCSE on several semantic textual similarity (STS) tasks from our reproducibilty study.


|        Model       |STS-12  | STS-13 | STS-14 | STS-15 |STS-16 | STS-B | SICK-R | Avg. |
|:--------------|:-------:|:-----:|:--------:|:---------:|:-------:|-------:|-------:|-------:|
| unsup-simcse-bert-base   | 65.84 | 78.96 | 71.87 | 80.28 | 77.34 | 75.50 | 71.37 | 74.45 |
| unsup-simcse-bert-large  | 46.00 | 65.41 | 53.54 | 67.32 | 69.84 | 55.50 | 65.21 | 60.40 |
| unsup-simcse-roberta-base | 67.05 | 79.54 | 72.13 | 79.81 | 77.94 | 80.41 | 70.89 | 75.40 |
| unsup-simcse-roberta-large | 65.10 | 79.76 | 71.97 | 81.92 | 77.60 | 78.54 | 69.11 | 74.86 |



