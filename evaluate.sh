#!/bin/bash

python evaluation.py     --model_name_or_path result/$1     --pooler cls     --task_set sts     --mode $2 > result/$1/eval_$2_results.txt
