# Basic configurations are defined here.
# includes json path, image path,

import os
import tensorflow as tf
# --------------------------------------------------------------------------------

# Three JSON files of are in the dataset directory
# os.path.join connect two paths
dataset_path = "dataset_hotdog"
train_jason_path = os.path.join(dataset_path, "transforms_train.json")
test_jason_path = os.path.join(dataset_path, "transforms_val.json")
val_jason_path = os.path.join(dataset_path, "transforms_test.json")

# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# define a output path, includes 2D images, 3D image, coarse model and fine model
nerf_output_path = "Nerf_output"
image_output_path = os.path.join(nerf_output_path, "2D images")
psnr_output_path = os.path.join(nerf_output_path, "psnr image")
video_file_path = os.path.join(nerf_output_path, "video")
video_output_path = os.path.join(video_file_path, "3D image.mp4")
coarse_save_path = os.path.join(nerf_output_path, "coarse model")
fine_save_path = os.path.join(nerf_output_path, "fine model")

# --------------------------------------------------------------------------------

# Effectively reduce manual parameter adjustment during debugging and optimization,
# and improve code portability and performance.
AUTO = tf.data.AUTOTUNE

# --------------------------------------------------------------------------------

# image size
image_width = 80
image_height = 80

# dimension for positional encoding
xyz_dim = 8
dir_dim = 4

# near and far boundary of the 3D scene
near_boundary = 2.0
far_boundary = 6.0

# number of samples for coarse and fine model
num_sample_coarse = 32 # 32
num_sample_fine = 68 # 70

# --------------------------------------------------------------------------------
"""
In one model training, the number of training samples processed at the same time
batch size bigger ---> training faster , may cause insufficient video memory
By testing, here batch size = 1 will be fine
"""
# batch size
batch_size = 1

# number of dense units of the layers
dense_unit = 82

# epoch , training epoch
epoch = 200

# step pre epoch(fit parameter)
# initial 30,5,120  psnr = 25.
step_per_epoch = 50

# --------------------------------------------------------------------------------

# 3D image parameters
fps = 33

#  1 <= video quality <= 10
video_quality = 9

# --------------------------------------------------------------------------------
