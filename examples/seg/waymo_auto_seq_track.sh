export CUDA_VISIBLE_DEVICES=0
# 定义场景数组
scenes=("0017085" "0145050" "0147030" "0158150")
cams=("1" "2")


# 遍历每个场景
for scene in "${scenes[@]}"; do
    for cam in "${cams[@]}"; do
        
        python image_sequence_tracking.py \
            --source /data/waymo/pvg_scenes/${scene}/image_${cam}\
            --output /data/waymo/pvg_scenes/${scene}/tracking_${cam}_vis \
            --conf-thres 0.8 \
            --device cuda:0 \
            --save-vid \
            --save-npy \
            --npy-path /data/waymo/pvg_scenes/${scene}/tracking_${cam} \

    done

done


