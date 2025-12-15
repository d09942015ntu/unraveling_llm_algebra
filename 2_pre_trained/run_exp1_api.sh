source venv/bin/activate

out_dir="output_biggsm"
mkdir -p ${out_dir}

types=(0_none 1_fw 2_bw 3_full 11_fwN 21_bwN 31_fullN)

DEFAULT_SEED=0

model_name="qwen/qwen-2.5-7b-instruct"
model_tag="qwen7b"

seed="${1:-$DEFAULT_SEED}"
echo "seed=${seed}"
for ttype in ${types[@]}; do
    python3 my_1_api.py \
        --model_name=${model_name} \
        --data_file="data/biggsm/data.jsonl" \
        --output_file="${out_dir}/${model_tag}_knowledge_${ttype}_${seed}.jsonl" \
        --knowledge_file="data/biggsm/knowledge_${ttype}.jsonl" \
        --seed=${seed}
done
