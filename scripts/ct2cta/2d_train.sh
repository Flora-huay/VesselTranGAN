NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 python pix2pix2d_attention_train.py --gpu_ids='0,1' \
																								--batch_size=2 \
																								--save_path="./experiment_0927_2dconv_attention_multiresolution" \
																								--latest_train_epoch=16 \
																								--lr_g=0.1e-6 \
																								--lr_d=0.1e-6