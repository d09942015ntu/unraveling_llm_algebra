#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.


source venv_llm_math/bin/activate

dataset_type='inv+invx+lh+rh'

DEFAULT_CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES="${1:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
# Export the CUDA_VISIBLE_DEVICES variable

export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"


model_name='gpt2'

datasets=(
7_100
7_1000
7_10000
7_300
7_3000
)
for ds in ${datasets[@]};do
    data_name=inv_64_${ds}
    python3 trainer.py  \
            --model_name=${model_name}  \
            --dataset_dir=./data/${data_name}  \
            --dataset_type=${dataset_type}  \
            --batch_size=768 \
            --rm_position=0 \
            --output_name=${data_name}
done

