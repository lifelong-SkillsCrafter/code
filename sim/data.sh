TASKS=(
    "open_drawer"
)

SAVE_PATH="data3/val"     
IMAGE_SIZE="128,128"
RENDERER="opengl"
EPISODES=800
PROCESSES=50
ALL_VARIATIONS="--all_variations=True"   


for task in "${TASKS[@]}"; do
    echo "=================================================================="
    echo " $(date '+%Y-%m-%d %H:%M:%S') 开始生成任务: ${task}"
    echo "=================================================================="

    python RLBench/tools/dataset_generator.py \
        --tasks "${task}" \
        --save_path "${SAVE_PATH}" \
        --image_size "${IMAGE_SIZE}" \
        --renderer "${RENDERER}" \
        --episodes_per_task "${EPISODES}" \
        --processes "${PROCESSES}" \
        ${ALL_VARIATIONS}

    if [ $? -ne 0 ]; then
        echo "[ERROR] 任务 ${task} 生成失败！详见日志: ${task_log}"
    else
        echo "[SUCCESS] 任务 ${task} 生成完成！"
    fi
done

echo "所有任务生成完成！最终数据保存在: ${SAVE_PATH}"