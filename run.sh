# CUDA_VISIBLE_DEVICES=1,2 \
# python train.py \
# --dataroot ./datasets \
# --name dcd_mtl \
# --gpu_ids 0,1 \
# --input_nc 1 \
# --output_nc 1 \
# --batch_size 1 \
# --phase train \
# --is_mtl

CUDA_VISIBLE_DEVICES=0 \
python test.py \
--results_dir ./results/dcd_mtl \
--name dcd_mtl \
--model_suffix _A \
--dataset_mode unaligned \
--gpu_ids 0 \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--phase test \
--dataroot ./datasets \
--is_mtl 