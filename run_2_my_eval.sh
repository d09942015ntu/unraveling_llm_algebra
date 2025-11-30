source venv/bin/activate

mkdir -p outputs_eval
types=(0_none 1_full 2_com 3_xop 4_noop)

for seed in $(seq 0 2); do
  for ttype in ${types[@]}; do
      python3 my_2_eval.py --input_file=outputs/qwen_knowledge_${ttype}_${seed}.jsonl > outputs_eval/qwen_eval_${ttype}_${seed}.txt
  done
done
