CUDA_VISIBLE_DEVICES=2 python pix2pix2d_attention_test.py --gpu_ids='0' \
														  --batch_size=1 \
														  --save_path="./experiment_0925_2dconv_attention_multiresolution" \
														  --nii_ct_root="./input_ct" \
														  --nii_cta_gt_root="./input_cta" \
														  --nii_cta_save_root="./output_cta" \
														  --latest_train_epoch=100