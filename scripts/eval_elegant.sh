logdirs=(
    "logs/diffuser_d4rl/hopper-medium-expert-v2/h_20-bs_64-r_400.0-guidew_1.2/100" \
    "logs/diffuser_d4rl/hopper-medium-expert-v2/h_20-bs_128-r_400.0-guidew_1.2/100" \
    # "logs/diffuser_d4rl/hopper-medium-expert-v2/h_20-bs_256-r_400.0-guidew_1.2/100" \
    # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_20-bs_64-r_400.0-guidew_1.2/100" \
    # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_20-bs_128-r_400.0-guidew_1.2/100" \
    # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_20-bs_256-r_400.0-guidew_1.2/100" \
    # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_100-r_400.0-guidew_1.2-ld_0.95/100"
)

commands=()
# 循环遍历logdirs列表
for logdir in "${logdirs[@]}"; do
    command="python evaluate.py $logdir --epochs 500 900"
    commands+=("$command")
done

declare -A running_commands

# 循环遍历命令队列，直到队列为空
for command in "${commands[@]}"; do
    # 使用nvidia-smi命令获取GPU信息
    gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader)

    # 使用循环遍历GPU信息
    while IFS=',' read -r gpu_index gpu_utilization memory_used memory_total; do
        if [[ -z "${running_commands[$gpu_index]}" ]]; then
            # 检查显存占用和GPU利用率是否都低于阈值（例如，10%）
            memory_threshold=10
            utilization_threshold=10

            memory_used="${memory_used%% *}"
            memory_total="${memory_total%% *}"
            memory_used_percent=$(echo "scale=2; $memory_used / $memory_total * 100" | bc)

            if (( memory_used < memory_threshold && gpu_utilization < utilization_threshold )); then
                # 执行命令
                eval "$command -g $gpu_index" &

                # 输出使用的GPU和命令
                echo "在GPU $gpu_index 上执行命令：$command"
                running_commands[$gpu_index]=$command
                break
            fi
        fi
    done <<< "$gpu_info"

    sleep 5
    # 检查正在执行的命令数组，移除已完成的命令
    for gpu_index in "${!running_commands[@]}"; do
        if ! pgrep -f "${running_commands[$gpu_index]}" > /dev/null; then
            unset running_commands[$gpu_index]
        fi
    done
done