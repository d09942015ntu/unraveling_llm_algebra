source venv/bin/activate

output_dir="outputs_analysis"
mkdir -p ${output_dir}
types=(0_none 1_full 2_com 3_ide 11_fullN 21_comN 31_ideN)

#types=(0_none) # 1_full 2_com 5_xop 4_noop)

seed=0

#qwen/qwen-2.5-72b-instruct
#qwen/qwen-2.5-7b-instruct
#meta-llama/llama-3.1-8b-instruct
#meta-llama/llama-3.1-70b-instruct

model_tag="qwen72b"
seed=0
for ttype in ${types[@]}; do


    python3 my_5_error_analysis.py \
          --knowledge_file="data/biggsm/knowledge_${ttype}.jsonl" \
          --input_file=outputs/${model_tag}_knowledge_${ttype}_${seed}.jsonl > ${output_dir}/${model_tag}_eval_${ttype}_${seed}.txt
done

