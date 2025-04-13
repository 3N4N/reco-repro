# python -m trainers.supervised_partial \
#   --dataset "cityscapes" \
#   --data-path "dataset/cityscapes" \
#   --label-ratio "p25" \
#   --seed 0 \
#   --val-interval 500 \
#   --model reconet 



# #!/bin/bash

MAX_JOBS=8

# Array of all parameter sets
params=(
  # Cityscapes dataset commands
  "--dataset cityscapes --data-path dataset/cityscapes --label-ratio p0 --seed 0 --val-interval 500 --model reconet --gpu 3"
  "--dataset cityscapes --data-path dataset/cityscapes --label-ratio p1 --seed 0 --val-interval 500 --model reconet --gpu 3"
  "--dataset cityscapes --data-path dataset/cityscapes --label-ratio p5 --seed 0 --val-interval 500 --model reconet --gpu 3"
  "--dataset cityscapes --data-path dataset/cityscapes --label-ratio p25 --seed 0 --val-interval 500 --model reconet --gpu 3"
  # Pascal dataset commands
  "--dataset pascal --data-path dataset/pascal --label-ratio p0 --seed 0 --val-interval 500 --model reconet --gpu 1"
  "--dataset pascal --data-path dataset/pascal --label-ratio p1 --seed 0 --val-interval 500 --model reconet --gpu 3"
  "--dataset pascal --data-path dataset/pascal --label-ratio p5 --seed 0 --val-interval 500 --model reconet --gpu 2"
  "--dataset pascal --data-path dataset/pascal --label-ratio p25 --seed 0 --val-interval 500 --model reconet --gpu 2"
)

# Launch jobs
for p in "${params[@]}"; do
  # Wait if we already have max jobs running
  while [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]]; do
    sleep 60
  done
  
  echo "Starting job with params: $p"
  python3 trainers/supervised_partial.py $p &
done

# Wait for all jobs to complete
wait

echo "All training jobs completed"
