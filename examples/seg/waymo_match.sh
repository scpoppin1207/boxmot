# scenes=("0017085" "0145050" "0147030" "0158150")
# cams=("0" "1" "2")


scenes=("0158150")
cams=("0" "1")

for scene in "${scenes[@]}"; do
    python match.py \
    --view_A ../../output/${scene}/cam_0 \
    --view_B ../../output/${scene}/cam_1 \
    --dynamic_list_A "8,9,10,13,24,39" \
    --dynamic_list_B "9,34,48" \
    --metric "euclidean" \
    # --use_ema
done

# scenes=("0158150")
# cams=("0" "1")
# --dynamic_list_A "8,9,10,13,24,39" \
# --dynamic_list_B "9,34,48" \

# scenes=("0158150")
# cams=("0" "2")
# --dynamic_list_A "8,9,10,13,24,39" \
# --dynamic_list_B "4,26" \


# scenes=("0017085")
# cams=("0" "1")
# --dynamic_list_A "1,13,22" \
# --dynamic_list_B "1,86" \

# scenes=("0147030")
# cams=("0" "1")
# --dynamic_list_A "1,6,11,20" \
# --dynamic_list_B "1,3,8" \