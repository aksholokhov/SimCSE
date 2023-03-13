## (Reproducibility study) SimCSE: Simple Contrastive Learning of Sentence Embeddings

In this repo, we reproduce the results for unsupervised SimCSE as part of our final project for [CSE517 Natural Language Processing (NLP) course](https://nasmith.github.io/NLP-winter23/) at the University of Washington (UW). The authors of this reproducibility study are [@aksholokhov](https://github.com/aksholokhov) and [@birajpandey](https://github.com/birajpandey).  

The original authors of the paper are Tianyu Gao, Xingcheng Yao, and Danqi Chen. 

Link to original repo: https://github.com/princeton-nlp/SimCSE 

Link to original paper: https://arxiv.org/abs/2104.08821


## Quick Links

  - [Final report](final-report)
  - [Compute Details](#compute-details)
  - [Dependencies](#dependencies)
  - [Data download](#data-download)
  - [Data preprocessing](#data-preprocess)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Pre-trained models](#pretrained)
  - [Additional experiments](#additional-exp)
  - [Table of Results](#main-results)

## Final report
To read the full reproducibility study, please check out the `SimCSE_final_project.pdf` in the repo.


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
For unsupervised SimCSE, the authors sample 1 million sentences from English Wikipedia. To download the corresponding dataset, run the shell script `data/downloadwiki.sh`


```bash
sh data/downloadwiki.sh
```


## Data preprocessing
The main training script `train.py` handles all the data pre-processing steps. Please read through the `DataTrainingArguments` class to find all the possible preprocessing args.


## Training
We provide example training scripts for both unsupervised and supervised SimCSE. 
In `run_unsup_example.sh`, we provide a single-GPU (or CPU) example for the unsupervised version, and in `run_sup_example.sh` we give a **multiple-GPU** example for the supervised version. Both scripts call `train.py` for training. 

The possible the arguments for training are the following:
* `--train_file`: Training file path. We support "txt" files (one line for one sentence) and "csv" files (2-column: pair data with no hard negative; 3-column: pair data with one corresponding hard negative instance). You can use our provided Wikipedia or NLI data, or you can use your own data with the same format.
* `--model_name_or_path`: Pre-trained checkpoints to start with. For now we support BERT-based models (`bert-base-uncased`, `bert-large-uncased`, etc.) and RoBERTa-based models (`RoBERTa-base`, `RoBERTa-large`, etc.).
* `--temp`: Temperature for the contrastive loss.
* `--pooler_type`: Pooling method. It's the same as the `--pooler_type` in the [evaluation part](#evaluation).
* `--mlp_only_train`: We have found that for unsupervised SimCSE, it works better to train the model with MLP layer but test the model without it. You should use this argument when training unsupervised SimCSE models.
* `--hard_negative_weight`: If using hard negatives (i.e., there are 3 columns in the training file), this is the logarithm of the weight. For example, if the weight is 1, then this argument should be set as 0 (default value).
* `--do_mlm`: Whether to use the MLM auxiliary objective. If True:
  * `--mlm_weight`: Weight for the MLM objective.
  * `--mlm_probability`: Masking rate for the MLM objective.

All the other arguments are standard Huggingface's `transformers` training arguments. Some of the often-used arguments are: `--output_dir`, `--learning_rate`, `--per_device_train_batch_size`. In our example scripts, we also set to evaluate the model on the STS-B development set (need to download the dataset following the [evaluation](#evaluation) section) and save the best checkpoint.


## Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on semantic textual similarity (STS) tasks and downstream transfer tasks. For STS tasks, our evaluation takes the "all" setting, and report Spearman's correlation. See [our paper](https://arxiv.org/pdf/2104.08821.pdf) (Appendix B) for evaluation details.

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

Then come back to the root directory, you can evaluate any `transformers`-based pre-trained models using our evaluation code. For example,
```bash
python evaluation.py \
    --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
    --pooler cls \
    --task_set sts \
    --mode test
```

Arguments for the evaluation script are as follows,

* `--model_name_or_path`: The name or path of a `transformers`-based pre-trained checkpoint. You can directly use the models in the above table, e.g., `princeton-nlp/sup-simcse-bert-base-uncased`.
* `--pooler`: Pooling method. Now we support
    * `cls` (default): Use the representation of `[CLS]` token. A linear+activation layer is applied after the representation (it's in the standard BERT implementation). If you use **supervised SimCSE**, you should use this option.
    * `cls_before_pooler`: Use the representation of `[CLS]` token without the extra linear+activation. If you use **unsupervised SimCSE**, you should take this option.
    * `avg`: Average embeddings of the last layer. If you use checkpoints of SBERT/SRoBERTa ([paper](https://arxiv.org/abs/1908.10084)), you should use this option.
    * `avg_top2`: Average embeddings of the last two layers.
    * `avg_first_last`: Average embeddings of the first and last layers. If you use vanilla BERT or RoBERTa, this works the best.
* `--mode`: Evaluation mode
    * `test` (default): The default test mode. To faithfully reproduce our results, you should use this option.
    * `dev`: Report the development set results. Note that in STS tasks, only `STS-B` and `SICK-R` have development sets, so we only report their numbers. It also takes a fast mode for transfer tasks, so the running time is much shorter than the `test` mode (though numbers are slightly lower).
    * `fasttest`: It is the same as `test`, but with a fast mode so the running time is much shorter, but the reported numbers may be lower (only for transfer tasks).
* `--task_set`: What set of tasks to evaluate on (if set, it will override `--tasks`)
    * `sts` (default): Evaluate on STS tasks, including `STS 12~16`, `STS-B` and `SICK-R`. This is the most commonly-used set of tasks to evaluate the quality of sentence embeddings.
    * `transfer`: Evaluate on transfer tasks.
    * `full`: Evaluate on both STS and transfer tasks.
    * `na`: Manually set tasks by `--tasks`.
* `--tasks`: Specify which dataset(s) to evaluate on. Will be overridden if `--task_set` is not `na`. See the code for a full list of tasks.



## Pre-trained models 
Currently, the work only supports checkpoints of pre-trained BERT-based models like [bert_base](https://huggingface.co/bert-base-uncased), [bert_large](https://huggingface.co/bert-large-uncased), [roberta_base](https://huggingface.co/roberta-base), and [roberta_large](https://huggingface.co/roberta-large) which can be downloaded from [hugging face](https://huggingface.co)


## Additional experiments
For reproducing data augmentations and the uniformity/alignment analysis from the original paper, check out the jupyter notebook in `cse517wi23_reproducibility_study/Reproducibility.ipynb`. 

To run unsup SimCSE on different data augmentation, first generate the augmented dataset using the jupyter notebook. Then, run the bash files corresponding to the specific data augmentation. For example, here we run unsup SimCSE bert_base_uncased model on wiki1M data augmented using delete-one-word .

```bash
sh run_delete_one_word.sh
```


## Table of Results
We show the performance of unsupervied SimCSE on several semantic textual similarity (STS) tasks from our reproducibilty study. Our results generally match that of the authors in the paper.


|        Model       |STS-12  | STS-13 | STS-14 | STS-15 |STS-16 | STS-B | SICK-R | Avg. |
|:--------------|:-------:|:-----:|:--------:|:---------:|:-------:|-------:|-------:|-------:|
| unsup-simcse-bert-base   | 65.84 | 78.96 | 71.87 | 80.28 | 77.34 | 75.50 | 71.37 | 74.45 |
| unsup-simcse-bert-large  | 46.00 | 65.41 | 53.54 | 67.32 | 69.84 | 55.50 | 65.21 | 60.40 |
| unsup-simcse-roberta-base | 67.05 | 79.54 | 72.13 | 79.81 | 77.94 | 80.41 | 70.89 | 75.40 |
| unsup-simcse-roberta-large | 65.10 | 79.76 | 71.97 | 81.92 | 77.60 | 78.54 | 69.11 | 74.86 |



