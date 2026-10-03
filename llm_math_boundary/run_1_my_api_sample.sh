source venv/bin/activate

mkdir -p outputs_eval
types=(0_none 1_full 2_com 5_xop 4_noop)

DEFAULT_SEED=1

seed="${1:-$DEFAULT_SEED}"
echo "seed=${seed}"
for ttype in ${types[@]}; do
    python3 my_1_api.py \
        --model_name="qwen/qwen-2.5-72b-instruct" \
        --data_file="data/biggsm_samples/data.jsonl" \
        --output_file="outputs/qwen72b_samples_knowledge_${ttype}_${seed}.jsonl" \
        --knowledge_file="data/biggsm_samples/knowledge_${ttype}.jsonl" \
        --seed=${seed}
done
