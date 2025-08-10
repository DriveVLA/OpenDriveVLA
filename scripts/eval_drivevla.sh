CKPT_PATH=$1
NUM_GPU=$2

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_PATH_CLEAN=$(basename ${CKPT_PATH})
LOG_RESULT_DIR="output/${CKPT_PATH_CLEAN}/${TIMESTAMP}"

mkdir -p ${LOG_RESULT_DIR}/log
mkdir -p ${LOG_RESULT_DIR}/results
EVAL_LOG_FILE="${LOG_RESULT_DIR}/log/eval.log"

echo ">>> Experiment Configuration:" | tee -a ${EVAL_LOG_FILE}
echo "- LOG_RESULT_DIR: ${LOG_RESULT_DIR}" | tee -a ${EVAL_LOG_FILE}
echo "- CKPT_PATH: ${CKPT_PATH}" | tee -a ${EVAL_LOG_FILE}
echo "----------------------------------------" | tee -a ${EVAL_LOG_FILE}

# ----------------------------------------------------------------

echo ">>> Inference ${CKPT_PATH}" | tee -a ${EVAL_LOG_FILE}

PLAN_CONV_PATH="${LOG_RESULT_DIR}/results/plan_conv.json"

    # -m debugpy --listen 6000 --wait-for-client \
PYTHONPATH="$(pwd)":$PYTHONPATH \
torchrun --nproc_per_node=${NUM_GPU} \
    drivevla/inference_drivevla.py \
    --num-workers 4 \
    --bf16 \
    --model-path ${CKPT_PATH} \
    --output ${PLAN_CONV_PATH} \
    2>&1 | tee -a ${EVAL_LOG_FILE}

echo ">>> Evaluating ${PLAN_CONV_PATH}..." | tee -a ${EVAL_LOG_FILE}

python drivevla/eval_drivevla.py \
    --output ${PLAN_CONV_PATH} \
    2>&1 | tee -a ${EVAL_LOG_FILE}
