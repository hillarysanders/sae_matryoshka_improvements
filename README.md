
```
python3 train.py \
  --recon_variant matryoshka \
  --sparsity batchtopk \
  --target_l0 40 \
  --n_latents 2048 \
  --batch_size 1 \
  --seq_len 128 \
  --num_steps 50
```

```
python3 toy_absorption.py \
  --n_latents 512 \
  --batch_size 256 \
  --num_steps 2000 \
  --lambda_base 1e-3 \
  --method baseline_l1

python3 toy_absorption.py \
  --run_name toy_calib_test \
  --out_dir runs \
  --num_steps 1000 \
  --batch_size 512 \
  --n_latents 1024 \
  --lambda_base 1e-3 \
  --calibrate_l0 \
  --target_l0 3.0 \
  --calib_rounds 8 \
  --calib_batches 5 \
  --calib_tol 0.25 \
  --log_every 200
```