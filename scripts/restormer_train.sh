

#train dim48
#python train_solar_restormer_spbbdm_preprocess_tb_samples.py \
#  --train_a /home/a/solarimage/train/A \
#  --train_b /home/a/solarimage/train/B \
#  --val_a /home/a/solarimage/val/A \
#  --val_b /home/a/solarimage/val/B \
#  --out_dir experiments/restormer_solar_spbbdm_preprocess_dim48 \
#  --image_size 256 \
#  --batch_size 1 \
#  --val_batch_size 1 \
#  --accumulate_grad_batches 4 \
#  --epochs 200 \
#  --max_steps 40000 \
#  --lr 2e-4 \
#  --beta1 0.9 \
#  --weight_decay 0.0 \
#  --dim 48 \
#  --save_steps 10000,20000,30000,40000 \
#  --save_every_epochs 30 \
#  --log_every_steps 100 \
#  --sample_every_steps 1000 \
#  --num_sample_images 4 \
#



#test
#  python test_solar_restormer_frozen.py \
#  --test_a /home/a/frozen-test-set/condition \
#  --test_b /home/a/frozen-test-set/ground_truth \
#  --checkpoint experiments/restormer_solar_spbbdm_preprocess_dim48/checkpoints/best.pth \
#  --out_dir experiments/restormer_solar_spbbdm_preprocess_dim48/test_on_frozen_best \
#  --dim 48 \
#  --match_mode name \
#  --save_input_gt \
#  --save_vis



