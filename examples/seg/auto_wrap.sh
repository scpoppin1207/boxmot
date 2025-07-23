export CUDA_VISIBLE_DEVICES=0
# 定义场景数组
scenes=("0017085" "0145050" "0147030" "0158150")
# scenes=("0158150")


# 遍历每个场景
for scene in "${scenes[@]}"; do

   
    # python image_pair_tracking.py \
    #     --source /home/scp_recon/eval_output/two_stage_${scene}/detection\
    #     --output wrap_${scene}_output \
    #     --device cuda:0 \
    #     --all_seq_mask_path /data/waymo/pvg_scenes/${scene}/tracking_0 \
    #     --calib_path /data/waymo/pvg_scenes/${scene}/calib \

    python vis_score.py \
        --source_file wrap_${scene}_output \

done


