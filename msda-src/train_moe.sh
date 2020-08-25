CUDA_VISIBLE_DEVICES=3 \
  python amazon-chen/senti_moe.py \
  --cuda \
  --dropout 0.0 \
  --train dvd,electronics,kitchen \
  --test books \
  --max_epoch 50 \
  --save_model dek-b.moe.ckpt \
  --n_d 500 \
  --lambda_moe 0.8 \
  --lambda_critic 0 \
  --lambda_entropy 0.1 \
  --batch_size 32 \
  --m_rank 50 \
  --metric mahalanobis \
  --activation relu

