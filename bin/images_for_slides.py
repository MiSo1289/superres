#!/usr/bin/env python3

import math
import os

import pyics
import napari
import numpy as np
import imageio
from matplotlib import pyplot as plt

results_root = '/home/miso/Projects/Courses/PV162/data/colon-tissue/out-4x'
images_root = '/home/miso/Projects/Courses/PV162/images'

gt = pyics.read_ics(
    os.path.join(results_root, 'ground_truth/image.ics'))

xz_inter = pyics.read_ics(
    os.path.join(results_root, 'interpolated/image.ics'))

xy_inter = pyics.read_ics(
    os.path.join(results_root, 'train/down-resampled/image.ics'))

inference = pyics.read_ics(
    os.path.join(results_root, 'inference/ssr-p32-40-epochs/image.ics'))

# downsampled = np.copy(gt)
# for i in range(gt.shape[0]):
#     if i % 4 != 0:
#         downsampled[i, :, :] = 0
#
# def save(seq):
#     max_val = -math.inf
#     min_val = math.inf
#     for _, im in seq:
#         max_val = max(max_val, np.max(im))
#         min_val = min(min_val, np.min(im))
#     for name, im in seq:
#         imageio.imwrite(os.path.join(images_root, f'{name}.png'),
#                         np.require(((im - min_val) / (max_val - min_val)) * 255,
#                                    dtype=np.uint8))
#
#
# for prefix, slice_size, off in (('', 512, 0), ('small_', 256, 0), ('tiny_', 128, 64)):
#     y_start = (gt.shape[1] - slice_size) // 2 + off
#     x_start = (gt.shape[2] - slice_size) // 2 + off
#     y_slice = gt.shape[1] // 2
#     z_slice = gt.shape[0] // 2
#
#     save([
#         (f'{prefix}downsampled_xz',
#          downsampled[:, y_slice, x_start:x_start + slice_size]),
#     ])
#     save([
#         (f'{prefix}gt_xz',
#          gt[:, y_slice, x_start:x_start + slice_size]),
#         (f'{prefix}inference_xz',
#          inference[:, y_slice, x_start:x_start + slice_size]),
#         (f'{prefix}inter_xz',
#          xz_inter[:, y_slice, x_start:x_start + slice_size]),
#     ])
#     save([
#         (f'{prefix}inference_diff_xz',
#          np.abs(np.require(gt[:, y_slice, x_start:x_start + slice_size],
#                            dtype=np.float32) -
#                 inference[:, y_slice, x_start:x_start + slice_size])),
#         (f'{prefix}inter_diff_xz',
#          np.abs(np.require(gt[:, y_slice, x_start:x_start + slice_size],
#                            dtype=np.float32) -
#                 xz_inter[:, y_slice, x_start:x_start + slice_size])),
#     ])
#     save([
#         (f'{prefix}gt_xy',
#          gt[z_slice, y_start:y_start + slice_size,
#          x_start:x_start + slice_size]),
#         (f'{prefix}inter_xy',
#          xy_inter[z_slice, y_start:y_start + slice_size,
#          x_start:x_start + slice_size]),
#     ])


# Histograms
inter_diff = np.abs(np.require(gt, dtype=np.int32)
                    - np.require(xz_inter, dtype=np.int32))
inference_diff = np.abs(np.require(gt, dtype=np.int32)
                        - np.require(inference, dtype=np.int32))
diff_min = 0
diff_max = max(np.max(inference_diff), np.max(inter_diff))
nbins = 100

bins = np.linspace(diff_min, diff_max, diff_max - diff_min + 1)

plt.hist(inter_diff.flatten(), bins, alpha=0.5, label='B-spline', color='c',
         density=True)
plt.hist(inference_diff.flatten(), bins, alpha=0.5, label='SMORE', color='m',
         density=True)
plt.legend(loc='upper right')
plt.xlim((diff_min, diff_max * 0.5))
plt.savefig(os.path.join(images_root, 'diff_hist.pdf'))
