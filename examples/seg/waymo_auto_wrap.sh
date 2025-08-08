export CUDA_VISIBLE_DEVICES=0
# 定义场景数组
# scenes=("0017085" "0145050" "0147030" "0158150")
scenes=("0017085")
cams=("0")



# 遍历每个场景
for scene in "${scenes[@]}"; do
    for cam in "${cams[@]}"; do
        python image_pair_tracking.py \
            --source /home/scp_recon/eval_output/two_stage_${scene}_3cam_all_points_init/detection\
            --cam ${cam} \
            --output wrap_${scene}_cam${cam}_output \
            --device cuda:0 \
            --all_seq_mask_path /data/waymo/pvg_scenes/${scene}/tracking_${cam} \
            --start_frame 0

        python vis_score.py \
            --source_file wrap_${scene}_cam${cam}_output
    done
done


