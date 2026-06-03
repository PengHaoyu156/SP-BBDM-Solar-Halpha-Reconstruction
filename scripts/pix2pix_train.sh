
#train
#python train.py \
#  --dataroot /home/a/solarimage_pix2pix/AB \
#  --name solar304_to_ha_pix2pix_standard \
#  --model pix2pix \
#  --dataset_mode aligned \
#  --direction AtoB \
#  --input_nc 3 \
#  --output_nc 3 \
#  --netG unet_256 \
#  --netD basic \
#  --norm batch \
#  --gan_mode vanilla \
#  --lambda_L1 100 \
#  --pool_size 0 \
#  --lr 0.0002 \
#  --beta1 0.5 \
#  --batch_size 1 \
#  --preprocess none \
#  --save_epoch_freq 10 \
#  --n_epochs 100 \
#  --n_epochs_decay 100

#test
#python test.py \
#  --dataroot /home/a/solarimage_pix2pix/AB \
#  --name solar304_to_ha_pix2pix_standard \
#  --model pix2pix \
#  --dataset_mode aligned \
#  --direction AtoB \
#  --input_nc 3 \
#  --output_nc 3 \
#  --netG unet_256 \
#  --norm batch \
#  --preprocess none \
#  --phase test \
#  --epoch latest \
#  --num_test 10000




