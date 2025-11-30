source venv/bin/activate

#types=(0_none 1_full 2_com)
types=(4_noop)
for ttype in ${types[@]}; do
    python3 my_eval.py --input_file=outputs/qwen_knowledge_${ttype}.jsonl > outputs_eval/eval_${ttype}.txt
done
