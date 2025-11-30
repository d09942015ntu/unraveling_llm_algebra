source venv/bin/activate

mkdir -p outputs_eval
types=(0_none 1_full 2_com 3_xop 4_noop)

for s in $(seq 0 2); do
  for ttype in ${types[@]}; do
      python3 my_1_api.py \
          --model_name="qwen/qwen-2.5-7b-instruct" \
          --data_file="data/biggsm/data.jsonl" \
          --output_file="outputs/qwen_knowledge_${ttype}.jsonl" \
          --knowledge_file="data/biggsm/knowledge_${ttype}_${seed}.jsonl" \
          --seed=0
  done
done