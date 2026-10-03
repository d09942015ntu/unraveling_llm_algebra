source venv/bin/activate

mkdir -p outputs_eval
#types=(11_fullN 22_comN 31_ideN)

types=(22_comN)

DEFAULT_SEED=0

#qwen/qwen-2.5-72b-instruct
#qwen/qwen-2.5-7b-instruct
#meta-llama/llama-3.1-8b-instruct
#meta-llama/llama-3.1-70b-instruct

model_name="qwen/qwen-2.5-7b-instruct"
model_tag="qwen7b"

seed="${1:-$DEFAULT_SEED}"
echo "seed=${seed}"
for ttype in ${types[@]}; do
    python3 my_1_api.py \
        --model_name=${model_name} \
        --data_file="data/biggsm/data.jsonl" \
        --output_file="outputs/${model_tag}_knowledge_${ttype}_${seed}.jsonl" \
        --knowledge_file="data/biggsm/knowledge_${ttype}.jsonl" \
        --seed=${seed}
done
