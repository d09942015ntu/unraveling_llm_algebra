#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.


source venv_llm_math/bin/activate

python3 dataset_generator.py

dataset_type='com+ide+comx+idex+lh+rh+z0'
DEFAULT_CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES="${1:-$DEFAULT_CUDA_VISIBLE_DEVICES}"

export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

model_name='gpt2'

datasets=(
7_100
7_300
11_100
11_300
13_100
13_300
)
for ds in ${datasets[@]};do
    data_name=quick_64_${ds}
    python3 trainer.py  \
            --model_name=${model_name}  \
            --dataset_dir=./data/${data_name}  \
            --dataset_type=${dataset_type}  \
            --batch_size=128 \
            --output_name=${data_name}_seqnew \
            --logging_step=50 \
            --num_train_epochs=10
done

python3 vis_plot_test_acc.py --data_prefix="quick_64"
python3 vis_plot_convergence.py  --data_prefix="quick_64" --n=7 --scale=300
python3 vis_com_std.py --data_prefix="quick_64"
python3 vis_ide_std.py --data_prefix="quick_64"

echo "Complete!"
