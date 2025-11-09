export CUDA_VISIBLE_DEVICES=0
# 定义场景数组
# scenes=("0001" "0002" "0006" )
# cams=("02" "03")
scenes=("0001")
cams=(0 1)



# 遍历每个场景
for scene in "${scenes[@]}"; do
    if [[ "$scene" == "0001" ]]; then
        start_frame=380
    else
        start_frame=65
    fi

    for cam in "${cams[@]}"; do
        output_path="kitti_${scene}_cam${cam}_output"
        python image_pair_tracking.py \
            --source /home/scp_recon/eval_output/kitti_${scene}/detection\
            --cam ${cam} \
            --conf-thres 0.5 \
            --output ${output_path} \
            --device cuda:0 \
            --all_seq_mask_path /data/kitti_pvg/training/tracking_data_0$((cam + 2))/${scene} \
            --start_frame ${start_frame} \
           

        python vis_score.py \
            --source_file ${output_path}
    done
done


