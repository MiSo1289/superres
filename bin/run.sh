#!/usr/bin/env bash

BIN_DIR="$(dirname "${0}")"
SRC_DIR="${BIN_DIR}/.."

export PATH="${PATH}:${BIN_DIR}"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}"

DATA_ROOT="/data/miso/PV162"
OUT_ROOT="${DATA_ROOT}/out"
IMG_NAME="image-final_0000.ics"
INFERENCE_AXIS=0
TRAIN_AXIS=2
AUX_AXIS=1
SLICE_SEPARATION=4
NUM_ROTATIONS=8
NUM_PARTITIONS=2
RGB_RANGE=600

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
superres rotate --axis-1 "${TRAIN_AXIS}" --axis-2 "${AUX_AXIS}" \
  -p "${NUM_PARTITIONS}" -n "${NUM_ROTATIONS}" -o "${OUT_ROOT}/train/rotated" \
  "${OUT_ROOT}/interpolated/"*
# Blur on the training axis
superres gaussian-blur -a "${TRAIN_AXIS}" --fwhm "${SLICE_SEPARATION}" \
  -p "${NUM_PARTITIONS}" -o "${OUT_ROOT}/train/blurred" \
  "${OUT_ROOT}/train/rotated/"*
# Introduce aliasing on the training axis
superres create-aliasing -a "${TRAIN_AXIS}" -x "${SLICE_SEPARATION}" \
  -p "${NUM_PARTITIONS}" -o "${OUT_ROOT}/train/aliased" \
  "${OUT_ROOT}/train/blurred/"*

echo "Step 3: train a SSR network"
superres train --infer-axis "${INFERENCE_AXIS}" -s "${SLICE_SEPARATION}" \
  -o "${OUT_ROOT}/model/ssr-cuda" -t "${OUT_ROOT}/train/rotated" \
  --rgb-range "${RGB_RANGE}" "${OUT_ROOT}/train/aliased/"*

echo "Step 4: run inference"
superres infer --train-axis "${TRAIN_AXIS}" -s "${SLICE_SEPARATION}" \
  -m "${OUT_ROOT}/model/ssr-cuda" -o "${OUT_ROOT}/inferrence/ssr" \
  --rgb-range "${RGB_RANGE}" "${OUT_ROOT}/anisotropic/"*
