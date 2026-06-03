

#
#python extract_bright_filament_rois.py \
#  --real_dir images/restormer/gt \
#  --fake_dir images/pix2pix/fake_B \
#  --out_dir /home/a/roi_eval_pix2pix/pix2pix \
#  --roi_size 48 \
#  --max_per_type 50 \
#  --match_mode name \
#  --inner_disk_ratio 0.80 \
#  --save_overlays \
#  --overwrite

#
python eval_roi_specific_metrics.py \
  --bright_real roi_eval_pix2pix/SP-BBDM-REF/bright/real \
  --bright_fake roi_eval_pix2pix/SP-BBDM-REF/bright/fake \
  --filament_real roi_eval_pix2pix/SP-BBDM-REF/filament/real \
  --filament_fake roi_eval_pix2pix/SP-BBDM-REF/filament/fake \
  --out_csv /home/a/roi_eval_pix2pix/SP-BBDM-REF/roi_specific_metrics.csv \
  --summary_txt /home/a/roi_eval_pix2pix/SP-BBDM-REF/roi_specific_metrics_summary.txt



