# MedicalProjects

## CT2CTA
    nii数据集目录结构
        dataset
            854985
                20211011
                    CP.nii.gz
                    CT.nii.gz
                20211012
                    CP.nii.gz
                    CT.nii.gz
                ....
            ....
    cd scripts/ct2cta
    1. 数据预处理
        python nii_preprocess_center_split_patch.py
        python nii_preprocess_center_split_patch_image.py
    2. 模型训练
        3D卷积:
            CUDA_VISIBLE_DEVICES=0,1,2,3 python pix2pix3d_train.py
            --gpu_ids='0,1,2,3' --center_depth=32 --batch_size=2
            --lambda_pixel=500 --use_amp=False --loss_type="l1" --use_gan_loss=False 
            --generator_scale=1.25 --use_checkpoint=True
            --save_path="./experiment_0911_nogan_gen1.25"
        2D卷积:
            CUDA_VISIBLE_DEVICES=0,1,2 python pix2pix2d_attention_train.py 
            --gpu_ids='0,1,2' --batch_size=2 --save_path="./experiment_0925_2dconv_attention_multiresolution"
    3. 模型测试
        3D卷积:
            CUDA_VISIBLE_DEVICES=5,6 python pix2pix3d_test.py 
                --gpu_ids='0,1' --center_depth=32 --batch_size=1
                --use_gan_loss=False --generator_scale=1.25
                --save_path="./experiment_0911_nogan_gen1.25"
                --nii_ct_root="./input_ct" --nii_cta_gt_root="./input_cta" 
                --nii_cta_save_root="./output_cta"
        2D卷积:
            CUDA_VISIBLE_DEVICES=5 python pix2pix2d_attention_test.py
                --gpu_ids='0' --batch_size=1 --save_path="./experiment_0925_2dconv_attention_multiresolution"
                --nii_ct_root="./input_ct" --nii_cta_gt_root="./input_cta" 
                --nii_cta_save_root="./output_cta"
