export CUDA_VISIBLE_DEVICES=0
# 定义场景数组
scenes=("0001" "0002" "0006" )
cams=("02" "03")


# 遍历每个场景
for scene in "${scenes[@]}"; do
    for cam in "${cams[@]}"; do
        
        python image_sequence_tracking.py \
            --source /data/kitti_pvg/training/image_${cam}/${scene}\
            --output /data/kitti_pvg/training/tracking_vis_${cam}/${scene} \
            --conf-thres 0.8 \
            --device cuda:0 \
            --save-vid \
            --save-npy \
            --npy-path /data/kitti_pvg/training/tracking_data_${cam}/${scene} \

    done

done


