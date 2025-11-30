source venv/bin/activate

mkdir -p outputs_eval
types=(0_none 1_full 2_com 3_xop 4_noop)

for ttype in ${types[@]}; do
    python3 my_2_eval.py --input_file=outputs/qwen_knowledge_${ttype}.jsonl > outputs_eval/eval_${ttype}.txt
done
