source venv/bin/activate

mkdir -p outputs_eval
types=(11_fwN 21_bwN 31_fullN 1_fw 2_bw 3_full 0_none) # 1_full 2_com 3_ide 11_fullN 21_comN 31_ideN)

DEFAULT_SEED=0

#qwen/qwen-2.5-72b-instruct
#qwen/qwen-2.5-7b-instruct
#meta-llama/llama-3.1-8b-instruct
#meta-llama/llama-3.1-70b-instruct

model_name="qwen/qwen-2.5-7b-instruct"
model_tag="qwen7b"
data_name="MAWPS"

seed="${1:-$DEFAULT_SEED}"
echo "seed=${seed}"
for ttype in ${types[@]}; do
    python3 my_1_api.py \
        --model_name=${model_name} \
        --data_file="data/${data_name}/set${seed}/data.jsonl" \
        --output_file="outputs/${model_tag}_${data_name}_${ttype}_${seed}.jsonl" \
        --knowledge_file="data/${data_name}/knowledge_${ttype}.jsonl" \
        --seed=${seed}
done
