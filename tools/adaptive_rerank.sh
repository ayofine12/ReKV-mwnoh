#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/root/mwnoh/ReKV-mwnoh"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROGRAM="${PROJECT_ROOT}/video_qa/rekv_offline_vqa.py"

export REKV_VIDEO_CACHE_DIR="${REKV_VIDEO_CACHE_DIR:-/mnt/ssd1/mwnoh/LVBench/video_cache}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/model:${PROJECT_ROOT}/model/longva:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAMPLE_FPS="${SAMPLE_FPS:-1}"
SAVE_DIR="${SAVE_DIR:-/mnt/ssd1/mwnoh/LVBench/results/retrieval_rerank}"
ANNO_PATH="${ANNO_PATH:-/mnt/ssd1/mwnoh/LVBench/data/video_info.json}"
MODEL="${MODEL:-llava_ov_7b}"
N_LOCAL="${N_LOCAL:-12544}"
RETRIEVE_SIZES="${RETRIEVE_SIZES:-32, 64, 96, 128}"
RERANK_CANDIDATE_MULTIPLIER="${RERANK_CANDIDATE_MULTIPLIER:-4}"
RETRIEVE_CHUNK_SIZE="${RETRIEVE_CHUNK_SIZE:-1}"
ENCODE_CHUNK_SIZE="${ENCODE_CHUNK_SIZE:-8}"
HEAD_SPECIFIC_RETRIEVAL="${HEAD_SPECIFIC_RETRIEVAL:-True}"
RETRIEVAL_FUSION="${RETRIEVAL_FUSION:-rerank}"
Q_TOKEN_AGG="${Q_TOKEN_AGG:-topk}"
Q_TOPK_RATIO="${Q_TOPK_RATIO:-0.3}"
K_TOKEN_AGG="${K_TOKEN_AGG:-topk}"
K_TOPK_RATIO="${K_TOPK_RATIO:-0.1}"
USE_VIDEO_CACHE="${USE_VIDEO_CACHE:-True}"
DEBUG_MODE="${DEBUG_MODE:-True}"

IFS=',' read -r -a RETRIEVE_SIZE_ARRAY <<< "${RETRIEVE_SIZES}"
RERANK_CANDIDATE_TOPKS_ARRAY=()
for retrieve_size in "${RETRIEVE_SIZE_ARRAY[@]}"; do
  trimmed_size="${retrieve_size//[[:space:]]/}"
  [[ -z "${trimmed_size}" ]] && continue
  RERANK_CANDIDATE_TOPKS_ARRAY+=("$((trimmed_size * RERANK_CANDIDATE_MULTIPLIER))")
done
RERANK_CANDIDATE_TOPKS="$(IFS=,; echo "${RERANK_CANDIDATE_TOPKS_ARRAY[*]}")"

ARGS=(
  --sample_fps "${SAMPLE_FPS}"
  --save_dir "${SAVE_DIR}"
  --anno_path "${ANNO_PATH}"
  --model "${MODEL}"
  --n_local "${N_LOCAL}"
  --retrieve_sizes "${RETRIEVE_SIZES}"
  --rerank_candidate_topks "${RERANK_CANDIDATE_TOPKS}"
  --retrieve_chunk_size "${RETRIEVE_CHUNK_SIZE}"
  --encode_chunk_size "${ENCODE_CHUNK_SIZE}"
  --head_specific_retrieval "${HEAD_SPECIFIC_RETRIEVAL}"
  --retrieval_fusion "${RETRIEVAL_FUSION}"
  --q_token_agg "${Q_TOKEN_AGG}"
  --q_topk_ratio "${Q_TOPK_RATIO}"
  --k_token_agg "${K_TOKEN_AGG}"
  --k_topk_ratio "${K_TOPK_RATIO}"
  --use_video_cache "${USE_VIDEO_CACHE}"
  --debug "${DEBUG_MODE}"
)

echo "============================================================"
echo "Running adaptive rerank configuration"
echo "Program: ${PROGRAM}"
echo "Save dir: ${SAVE_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Retrieve sizes: ${RETRIEVE_SIZES}"
echo "Rerank candidate topks: ${RERANK_CANDIDATE_TOPKS}"
echo "============================================================"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" "${PROGRAM}" "${ARGS[@]}" "$@"
