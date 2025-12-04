source venv/bin/activate

mkdir -p outputs_eval
types=(0_none 1_full 2_com 3_xop 4_noop)

#types=(0_none) # 1_full 2_com 3_xop 4_noop)

seed=0
#for seed in $(seq 0 2); do
  for ttype in ${types[@]}; do
      python3 my_2_eval.py --input_file=outputs/llama8b_knowledge_${ttype}_${seed}.jsonl > outputs_eval/llama8b_eval_${ttype}_${seed}.txt
  done
#done

for ttype in ${types[@]}; do
  echo "${ttype}"
  grep "averaged_correct:\([0-1].[0-9]\+\)" outputs_eval/llama8b_eval_${ttype}_* | grep "[0-1].[0-9]\+" -o | datamash mean 1
done


# Qwen7b
#0_none
#0.59071038251366
#1_full
#0.89672131147541
#2_com
#0.86994535519126
#3_xop
#0.72404371584699
#4_noop
#0.81803278688525
