source venv/bin/activate

mkdir -p outputs_eval
types=(0_none 1_full 2_com 3_ide 11_fullN 21_comN 31_ideN)
Tname=(None Orig Com Ide Orig-N Com-N Ide-N)

api_result_dir="output_biggsm"

seed=0

model_tag="qwen7b"

for seed in $(seq 0 2); do
  for ttype in ${types[@]}; do
      python3 my_2_eval.py --input_file=${api_result_dir}/${model_tag}_knowledge_${ttype}_${seed}.jsonl > outputs_eval/${model_tag}_eval_${ttype}_${seed}.txt
  done
done

echo "${model_tag}"
tidx=0
for ttype in ${types[@]}; do
  t_mean=$(grep "averaged_correct:\([0-1].[0-9]\+\)" outputs_eval/${model_tag}_eval_${ttype}_* | grep "[0-1].[0-9]\+" -o | datamash mean 1)
  t_std=$(grep "averaged_correct:\([0-1].[0-9]\+\)" outputs_eval/${model_tag}_eval_${ttype}_* | grep "[0-1].[0-9]\+" -o | datamash sstdev 1)
  t3_std=$(bc <<< "3*${t_std}")
  echo "(${Tname[${tidx}]}, ${t_mean})  +- (0.0, ${t3_std})"
  tidx=$(($tidx+1))
done
