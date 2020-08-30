CUDA_VISIBLE_DEVICES=3 \
  python amazon-chen/senti_unified.py \
  --cuda \
  --dropout 0.1 \
  --encoder mlp \
  --train dvd,electronics,kitchen \
  --test books \
  --max_epoch 15 \
  --save_model dek-b.unified.ckpt \
  --n_d 500 \
  --lambda_critic 0 \
  --batch_size 64 \
  --batch_size_d 64 \
  --activation relu

