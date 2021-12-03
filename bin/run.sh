#!/usr/bin/env bash

BIN_DIR="$(dirname "${0}")"
SRC_DIR="${BIN_DIR}/.."

export PATH="${PATH}:${BIN_DIR}"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}"

# Axis indexing: z=0, y=1, x=2
INFERENCE_AXIS=0
TRAIN_AXIS_1=2
TRAIN_AXIS_2=1
SLICE_SEPARATION=4
NUM_ROTATIONS=4
NUM_PARTITIONS=1
MAX_COLOR_LEVEL=600
TRAIN_EPOCHS=20
TRAIN_BATCH_SIZE=32
TRAIN_LEARNING_RATE="1e-3"
TRAIN_PATCH_SIZE_SSR=128
COMPARE_METRIC="mse"
INFER_CHUNK_SIZE=3

DATA_ROOT="/home/miso/Projects/Courses/PV162/data"
OUT_ROOT="${DATA_ROOT}/out-${SLICE_SEPARATION}x-no-saa"
IMG_NAME="image-final_0000.ics"
MODEL_NAME="ssr"

echo "Prepare ground truth by making the input data evenly divisible by slice separation"
superres slice -a "${INFERENCE_AXIS}" -s 0 -e 128 \
  -o "${OUT_ROOT}/ground_truth" \
  "${DATA_ROOT}/orig/${IMG_NAME}"

echo "Step 0: construct synthetic anisotropic data"
# Downsample original isotropic data to create fake anisotropic data
superres downsample -a "${INFERENCE_AXIS}" -x "${SLICE_SEPARATION}" \
  -p "${NUM_PARTITIONS}" -o "${OUT_ROOT}/anisotropic" \
  "${OUT_ROOT}/ground_truth/"*

echo "Step 1: preprocessing"
# Interpolate the anisotropic data to make them isotropic
superres bsp-interpolate -a "${INFERENCE_AXIS}" -x "${SLICE_SEPARATION}" \
  -p "${NUM_PARTITIONS}" -o "${OUT_ROOT}/interpolated" \
  "${OUT_ROOT}/anisotropic/"*

echo "Step 2: construct training data"
# Generate rotations around the inference axis
superres rotate --axis-1 "${TRAIN_AXIS_1}" --axis-2 "${TRAIN_AXIS_2}" \
  -p "${NUM_PARTITIONS}" -n "${NUM_ROTATIONS}" -o "${OUT_ROOT}/train/rotated" \
  "${OUT_ROOT}/interpolated/"*
# Introduce aliasing on the training axis
superres create-aliasing -a "${TRAIN_AXIS_1}" -x "${SLICE_SEPARATION}" \
  -p "${NUM_PARTITIONS}" -o "${OUT_ROOT}/train/down-resampled" \
  "${OUT_ROOT}/train/rotated/"*

echo "Step 4: train a SSR network"
superres train --in-memory --infer-axis "${INFERENCE_AXIS}" \
  --patch-size "${TRAIN_PATCH_SIZE_SSR}" -o "${OUT_ROOT}/model/${MODEL_NAME}.pt" -t "${OUT_ROOT}/train/rotated" \
  --max-color-level "${MAX_COLOR_LEVEL}" --epochs "${TRAIN_EPOCHS}" --batch-size "${TRAIN_BATCH_SIZE}" \
  --learning-rate "${TRAIN_LEARNING_RATE}" "${OUT_ROOT}/train/down-resampled/"*

echo "Step 6: run SSR inference"
superres infer --train-axis "${TRAIN_AXIS_1}" --infer-axis "${INFERENCE_AXIS}" \
  -m "${OUT_ROOT}/model/${MODEL_NAME}.pt" -o "${OUT_ROOT}/inference/${MODEL_NAME}" \
  --max-color-level "${MAX_COLOR_LEVEL}" "${OUT_ROOT}/interpolated/"*

echo "Step 7: compare the results to ground truth"
superres compare -m "${COMPARE_METRIC}" -r "${OUT_ROOT}/ground_truth/${IMG_NAME}" \
  "${OUT_ROOT}/interpolated"/* \
  "${OUT_ROOT}/inference/${MODEL_NAME}"/*