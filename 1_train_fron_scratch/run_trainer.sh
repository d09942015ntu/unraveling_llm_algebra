#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.


source venv_llm_math/bin/activate


dataset_type='com+ide+comx+idex+lh+rh+z0'


DEFAULT_CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES="${1:-$DEFAULT_CUDA_VISIBLE_DEVICES}"

export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

model_name='gpt2'

datasets=(
7_100
7_300
7_1000
7_3000
7_10000
11_100
11_300
11_1000
11_3000
11_10000
11_30000
13_100
13_300
13_1000
13_3000
13_10000
13_30000
)
for ds in ${datasets[@]};do
    data_name=all_64_${ds}
    python3 trainer.py  \
            --model_name=${model_name}  \
            --dataset_dir=./data/${data_name}  \
            --dataset_type=${dataset_type}  \
            --batch_size=768 \
            --output_name=${data_name}_seqnew
done

