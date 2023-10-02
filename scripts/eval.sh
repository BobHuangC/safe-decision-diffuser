logdirs=(
    # "logs/diffuser_d4rl/hopper-medium-expert-v2/h_20-bs_64-r_400.0-guidew_1.2/100" \
    # "logs/diffuser_d4rl/hopper-medium-expert-v2/h_20-bs_128-r_400.0-guidew_1.2/100" \

    # "logs/diffuser_d4rl/hopper-medium-expert-v2/h_20-bs_256-r_400.0-guidew_1.2/100" \
    # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_20-bs_64-r_400.0-guidew_1.2/100" \

    # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_20-bs_128-r_400.0-guidew_1.2/100" \
    # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_20-bs_256-r_400.0-guidew_1.2/100" \

    # "logs/diffuser_inv_d4rl/hopper-medium-expert-v2/h_100-r_400.0-guidew_1.2-ld_0.95/100" \

    "logs/diffuser_inv_d4rl_ema/hopper-medium-expert-v2/h_20-r_400.0-guidew_1.2-dstep_100/100" \
    "logs/diffuser_inv_d4rl_ema/hopper-medium-expert-v2/h_20-r_400.0-guidew_1.2-dstep_200/100" \
)

while getopts ":g:" opt; do
  case $opt in
    g)
      gpu_index=$OPTARG
      ;;
    \?)
      echo "Unknown options: -$OPTARG" >&2
      ;;
  esac
done

# 循环遍历logdirs列表
for logdir in "${logdirs[@]}"; do
    eval $"python evaluate.py $logdir --epochs 100 -g $gpu_index"
done