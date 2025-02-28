run_adv_values=( $(seq 0.1 0.1 0.9) )
for run_adv in "${run_adv_values[@]}"; do
  echo "$run_adv"
  echo "1"
done
