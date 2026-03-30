#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/root/mwnoh/ReKV-mwnoh"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROGRAM="${PROJECT_ROOT}/video_qa/rekv_offline_vqa.py"

export REKV_VIDEO_CACHE_DIR="${REKV_VIDEO_CACHE_DIR:-/mnt/ssd1/mwnoh/LVBench/video_cache}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/model:${PROJECT_ROOT}/model/longva:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAMPLE_FPS="${SAMPLE_FPS:-1}"
SAVE_DIR="${SAVE_DIR:-/mnt/ssd1/mwnoh/LVBench/results_event}"
ANNO_PATH="${ANNO_PATH:-/mnt/ssd1/mwnoh/LVBench/data/event.json}"
MODEL="${MODEL:-llava_ov_7b}"
N_LOCAL="${N_LOCAL:-6272}"
RETRIEVE_SIZE="${RETRIEVE_SIZE:-32}"
RETRIEVE_CHUNK_SIZE="${RETRIEVE_CHUNK_SIZE:-1}"
ENCODE_CHUNK_SIZE="${ENCODE_CHUNK_SIZE:-8}"
USE_VIDEO_CACHE="${USE_VIDEO_CACHE:-True}"
DEBUG_MODE="${DEBUG_MODE:-False}"

HEAD_SPECIFIC_FILTER="${HEAD_SPECIFIC_FILTER:-all}"   # all|true|false
RUN_TAG_FILTER="${RUN_TAG_FILTER:-}"                 # substring filter, optional
START_TAG="${START_TAG:-}"                           # exact tag to start from, optional
START_HEAD_SPECIFIC="${START_HEAD_SPECIFIC:-}"       # True|False, optional

BASE_ARGS=(
  --sample_fps "${SAMPLE_FPS}"
  --save_dir "${SAVE_DIR}"
  --anno_path "${ANNO_PATH}"
  --model "${MODEL}"
  --n_local "${N_LOCAL}"
  --retrieve_size "${RETRIEVE_SIZE}"
  --retrieve_chunk_size "${RETRIEVE_CHUNK_SIZE}"
  --encode_chunk_size "${ENCODE_CHUNK_SIZE}"
  --use_video_cache "${USE_VIDEO_CACHE}"
  --debug "${DEBUG_MODE}"
)

RUNS=(
  "mean_mean::--kv_repr mean --q_repr mean"
  "mean_token_qtopk::--kv_repr mean --q_repr token --q_token_agg topk --q_topk_ratio 0.3"
  "token_mean_kmax::--kv_repr token --q_repr mean --k_token_agg max"
  "token_mean_ktopk::--kv_repr token --q_repr mean --k_token_agg topk --k_topk_ratio 0.1"
  "token_token_kmax_qmean::--kv_repr token --q_repr token --k_token_agg max --q_token_agg mean"
  "token_token_kmax_qtopk::--kv_repr token --q_repr token --k_token_agg max --q_token_agg topk --q_topk_ratio 0.3"
  "token_token_ktopk_qmean::--kv_repr token --q_repr token --k_token_agg topk --k_topk_ratio 0.1 --q_token_agg mean"
  "token_token_ktopk_qtopk::--kv_repr token --q_repr token --k_token_agg topk --k_topk_ratio 0.1 --q_token_agg topk --q_topk_ratio 0.3"
)

should_run_head_specific() {
  local head_specific="$1"
  case "${HEAD_SPECIFIC_FILTER}" in
    all) return 0 ;;
    true) [[ "${head_specific}" == "True" ]] ;;
    false) [[ "${head_specific}" == "False" ]] ;;
    *)
      echo "Unsupported HEAD_SPECIFIC_FILTER: ${HEAD_SPECIFIC_FILTER}. Use all|true|false." >&2
      exit 1
      ;;
  esac
}

should_run_tag() {
  local tag="$1"
  if [[ -z "${RUN_TAG_FILTER}" ]]; then
    return 0
  fi
  [[ "${tag}" == *"${RUN_TAG_FILTER}"* ]]
}

should_start_at() {
  local tag="$1"
  local head_specific="$2"

  if [[ -z "${START_TAG}" ]]; then
    return 0
  fi

  if [[ "${tag}" != "${START_TAG}" ]]; then
    return 1
  fi

  if [[ -z "${START_HEAD_SPECIFIC}" ]]; then
    return 0
  fi

  [[ "${head_specific}" == "${START_HEAD_SPECIFIC}" ]]
}

run_one() {
  local tag="$1"
  local head_specific="$2"
  shift 2

  if ! should_run_head_specific "${head_specific}"; then
    return 0
  fi
  if ! should_run_tag "${tag}"; then
    return 0
  fi

  echo "============================================================"
  echo "Running tag=${tag} head_specific=${head_specific}"
  echo "============================================================"

  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" "${PROGRAM}" \
      "${BASE_ARGS[@]}" \
      --head_specific_retrieval "${head_specific}" \
      "$@"
  )
}

start_reached=0

for run_spec in "${RUNS[@]}"; do
  tag="${run_spec%%::*}"
  arg_string="${run_spec#*::}"
  read -r -a run_args <<< "${arg_string}"

  if [[ "${start_reached}" -eq 0 ]] && should_start_at "${tag}" "False"; then
    start_reached=1
  fi
  if [[ "${start_reached}" -eq 1 ]]; then
    run_one "${tag}" "False" "${run_args[@]}"
  fi

  if [[ "${start_reached}" -eq 0 ]] && should_start_at "${tag}" "True"; then
    start_reached=1
  fi
  if [[ "${start_reached}" -eq 1 ]]; then
    run_one "${tag}" "True" "${run_args[@]}"
  fi
done
