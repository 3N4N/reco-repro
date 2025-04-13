python -m trainers.supervised_full \
  --dataset "cityscapes" \
  --data-path "dataset/cityscapes" \
  --num-labeled 150 \
  --val-interval 5000 \
  --model reconet
