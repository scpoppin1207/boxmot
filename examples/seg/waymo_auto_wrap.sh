export CUDA_VISIBLE_DEVICES=4
# 定义场景数组
# scenes=("0017085" "0145050" "0147030" "0158150")
scenes=("0145050")
cams=("0")



# 遍历每个场景
for scene in "${scenes[@]}"; do
    for cam in "${cams[@]}"; do
        # python image_pair_tracking.py \
        #     --source /home/scp_recon/eval_output/${scene}_3cam_stage_1/detection\
        #     --cam ${cam} \
        #     --output wrap_${scene}_cam${cam}_output_box_new \
        #     --device cuda:0 \
        #     --conf-thres 0.8 \
        #     --all_seq_mask_path /data/waymo/pvg_scenes/${scene}/tracking_${cam} \
        #     --start_frame 0

        # python vis_score_ablation.py \
        #     --source_file wrap_${scene}_cam${cam}_output \
        #     --exp1_file wrap_${scene}_cam${cam}_output_box_motion_only \
        #     --exp2_file wrap_${scene}_cam${cam}_output_appearance_only

        python vis_score.py \
            --source_file wrap_${scene}_cam${cam}_output_box_new \
            --star_ids "7,2,5,1"

    done
done


# 