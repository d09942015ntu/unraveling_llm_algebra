source venv/bin/activate

mkdir -p outputs_eval
types=(0_none 3_full 1_fw 2_bw 31_fullN 11_fwN 21_bwN) #2_com 3_ide)
Tname=(None One FW BW One-N FW-N BW-N)

#types=(0_none) # 1_full 2_com 5_xop 4_noop)

seed=0

#qwen/qwen-2.5-72b-instruct
#qwen/qwen-2.5-7b-instruct
#meta-llama/llama-3.1-8b-instruct
#meta-llama/llama-3.1-70b-instruct

api_result_dir="output_MAWPS_2"

model_tag="qwen7b"

model_tags=(qwen7b qwen72b llama8b llama70b)

for model_tag in ${model_tags[@]};do
    for seed in $(seq 0 2); do
      for ttype in ${types[@]}; do
          python3 my_2_eval.py --input_file=${api_result_dir}/${model_tag}_MAWPS_${ttype}_${seed}.jsonl > outputs_eval/${model_tag}_MAWPS_eval_${ttype}_${seed}.txt
      done
    done

    echo "${model_tag}"
    tidx=0
    for ttype in ${types[@]}; do
      t_mean=$(grep "averaged_correct:\([0-1].[0-9]\+\)" outputs_eval/${model_tag}_MAWPS_eval_${ttype}_* | grep "[0-1].[0-9]\+" -o | datamash mean 1)
      t_var=$(grep "averaged_correct:\([0-1].[0-9]\+\)" outputs_eval/${model_tag}_MAWPS_eval_${ttype}_* | grep "[0-1].[0-9]\+" -o | datamash pvar 1)
      if [[ $t_var -le 0.0001 ]]; then
            echo "${t_var} less than 0.0001"
      fi
      echo "(${Tname[${tidx}]}, ${t_mean}) std: ${t_var}" 
      tidx=$(($tidx+1))
    done
    echo ""
done

# - ${t_var}"
# Qwen7b
#0_none
#0.59071038251366
#1_full
#0.89672131147541
#2_com
#0.86994535519126
#5_xop
#0.72404371584699
#4_noop
#0.81803278688525
