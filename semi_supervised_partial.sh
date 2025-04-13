# python -m trainers.semi_supervised_partial \
#   --dataset "cityscapes" \
#   --data-path "dataset/cityscapes" \
#   --label-ratio "p0" \
#   --seed 0 \
#   --val-interval 500 \
#   --model reconet \
#   --mixing-strategy classmix \
#   --disable-saving

#!/bin/bash

MAX_JOBS=8

# Array of all parameter sets
params=(
  # Cityscapes dataset commands
  "--dataset cityscapes --data-path dataset/cityscapes --label-ratio p0 --seed 0 --val-interval 500 --model reconet --mixing-strategy classmix "
  "--dataset cityscapes --data-path dataset/cityscapes --label-ratio p1 --seed 0 --val-interval 500 --model reconet --mixing-strategy classmix "
  "--dataset cityscapes --data-path dataset/cityscapes --label-ratio p5 --seed 0 --val-interval 500 --model reconet --mixing-strategy classmix "
  "--dataset cityscapes --data-path dataset/cityscapes --label-ratio p25 --seed 0 --val-interval 500 --model reconet --mixing-strategy classmix "
  # Pascal dataset commands
  "--dataset pascal --data-path dataset/pascal --label-ratio p0 --seed 0 --val-interval 500 --model reconet --mixing-strategy classmix "
  "--dataset pascal --data-path dataset/pascal --label-ratio p1 --seed 0 --val-interval 500 --model reconet --mixing-strategy classmix "
  "--dataset pascal --data-path dataset/pascal --label-ratio p5 --seed 0 --val-interval 500 --model reconet --mixing-strategy classmix "
  "--dataset pascal --data-path dataset/pascal --label-ratio p25 --seed 0 --val-interval 500 --model reconet --mixing-strategy classmix "
)

# Launch jobs
for p in "${params[@]}"; do
  # Wait if we already have max jobs running
  while [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]]; do
    sleep 60
  done
  
  echo "Starting job with params: $p"
  python3 trainers/semi_supervised_partial.py $p &
done

# Wait for all jobs to complete
wait

echo "All training jobs completed"
