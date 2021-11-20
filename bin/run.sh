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
NUM_ROTATIONS=8
NUM_PARTITIONS=1
RGB_RANGE=600
TRAIN_EPOCHS=16
TRAIN_BATCH_SIZE=64
TRAIN_LEARNING_RATE="1e-3"
TRAIN_PATCH_SIZE=128
COMPARE_METRIC="mse"

DATA_ROOT="/home/miso/Projects/Courses/PV162/data"
OUT_ROOT="${DATA_ROOT}/out-${SLICE_SEPARATION}x"
IMG_NAME="image-final_0000.ics"

echo "Step 0: construct synthetic anisotropic data"
# Downsample original isotropic data to create fake anisotropic data
superres downsample -a "${INFERENCE_AXIS}" -x "${SLICE_SEPARATION}" \
  -p "${NUM_PARTITIONS}" -o "${OUT_ROOT}/anisotropic" \
  "${DATA_ROOT}/orig/${IMG_NAME}"

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
# Blur on the training axis
superres gaussian-blur -a "${TRAIN_AXIS_1}" --fwhm "${SLICE_SEPARATION}" \
  -p "${NUM_PARTITIONS}" -o "${OUT_ROOT}/train/blurred" \
  "${OUT_ROOT}/train/rotated/"*
# Introduce aliasing on the training axis
superres create-aliasing -a "${TRAIN_AXIS_1}" -x "${SLICE_SEPARATION}" \
  -p "${NUM_PARTITIONS}" -o "${OUT_ROOT}/train/aliased" \
  "${OUT_ROOT}/train/blurred/"*

echo "Step 3: train a SSA network"
superres train --in-memory --infer-axis "${INFERENCE_AXIS}" -s "${SLICE_SEPARATION}" \
  --patch-size "${TRAIN_PATCH_SIZE}" -o "${OUT_ROOT}/model/ssa.pt" -t "${OUT_ROOT}/train/blurred" \
  --rgb-range "${RGB_RANGE}" --epochs "${TRAIN_EPOCHS}" --batch-size "${TRAIN_BATCH_SIZE}" \
  --learning-rate "${TRAIN_LEARNING_RATE}" "${OUT_ROOT}/train/aliased/"*

echo "Step 4: train a SSR network"
superres train --in-memory --infer-axis "${INFERENCE_AXIS}" -s "${SLICE_SEPARATION}" \
  --patch-size "${TRAIN_PATCH_SIZE}" -o "${OUT_ROOT}/model/ssr.pt" -t "${OUT_ROOT}/train/rotated" \
  --rgb-range "${RGB_RANGE}" --epochs "${TRAIN_EPOCHS}" --batch-size "${TRAIN_BATCH_SIZE}" \
  --learning-rate "${TRAIN_LEARNING_RATE}" "${OUT_ROOT}/train/aliased/"*

echo "Step 5: run SSA inference"
superres infer --train-axis "${TRAIN_AXIS_2}" --infer-axis "${INFERENCE_AXIS}" \
  -s "${SLICE_SEPARATION}" \
  -m "${OUT_ROOT}/model/ssa.pt" -o "${OUT_ROOT}/inference/ssa" \
  --rgb-range "${RGB_RANGE}" "${OUT_ROOT}/interpolated/"*

echo "Step 6: run SSR inference"
superres infer --train-axis "${TRAIN_AXIS_1}" --infer-axis "${INFERENCE_AXIS}" \
  -s "${SLICE_SEPARATION}" \
  -m "${OUT_ROOT}/model/ssr.pt" -o "${OUT_ROOT}/inference/ssr" -d \
  --rgb-range "${RGB_RANGE}" "${OUT_ROOT}/inference/ssa/"*

echo "Step 6.b: run SSR inference without SSA"
superres infer --train-axis "${TRAIN_AXIS_1}" --infer-axis "${INFERENCE_AXIS}" \
  -s "${SLICE_SEPARATION}" \
  -m "${OUT_ROOT}/model/ssr.pt" -o "${OUT_ROOT}/inference/ssr-only" -d \
  --rgb-range "${RGB_RANGE}" "${OUT_ROOT}/interpolated/"*

echo "Step 7: compare the results to ground truth"
superres compare -m "${COMPARE_METRIC}" -r "${DATA_ROOT}/orig/${IMG_NAME}" \
  "${OUT_ROOT}/interpolated"/* \
  "${OUT_ROOT}/inference/ssa"/* \
  "${OUT_ROOT}/inference/ssr"/* \
  "${OUT_ROOT}/inference/ssr-only"/*
