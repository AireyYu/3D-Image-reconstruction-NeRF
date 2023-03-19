"""
Nerf MEAN module, contains all configurations,parameters
after constructing all related modules, instantiate functions, models
This module is going to combine them together

1. Obtain jason data ---->2. Get Image path, c2w matrix ---->3. Get real Image(80x80) form image dataset---->
4. Get focal length from jason ---->5.Get rays form c2w matrix to form rays dataset ---->6. combine image and ray dataset ---->
7. Get models ---->8. compile ---> 9.define image plot callback ----> 10. fit model ----> 11. return coarse and fine models

"""

# import packages
import tensorflow as tf
from NERF3D.Nerf_data import read_json
from NERF3D.Nerf_data import get_imagepath_c2w
from NERF3D.Nerf_data import GetImages
from NERF3D.Nerf_data import GetRays
from NERF3D.Nerf_data import camera_focal
from NERF3D.Render_image import render_image_function
from NERF3D.hiera_sampling import hierarchical_sample
from NERF3D.Encoder import encoder
from NERF3D.Nerf_model import MLP_model
from NERF3D.Nerf_trainer import trainer
from NERF3D.Nerf_trainer import plotting_epoch_img
from NERF3D import parameter
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os

# -------------------------------------------------------------------------------------
# Make sure image output path exist
if not os.path.exists(parameter.image_output_path):
	os.makedirs(parameter.image_output_path)

if not os.path.exists(parameter.psnr_output_path):
	os.makedirs(parameter.psnr_output_path)
# -------------------------------------------------------------------------------------

# obtain all jason data
train_jason_data = read_json(parameter.train_jason_path)
test_jason_data = read_json(parameter.test_jason_path)
val_jason_data = read_json(parameter.val_jason_path)

# -------------------------------------------------------------------------------------

# from get_image_c2w , output an image path(where the images are saved) and camera to world matrix
train_image_path, train_c2w_matrix = get_imagepath_c2w(jsondata=train_jason_data, datasetPath=parameter.dataset_path)
test_image_path, test_c2w_matrix = get_imagepath_c2w(jsondata=test_jason_data, datasetPath=parameter.dataset_path)
val_image_path, val_c2w_matrix = get_imagepath_c2w(jsondata=val_jason_data, datasetPath=parameter.dataset_path)

# -------------------------------------------------------------------------------------

# Call the GetImage function, form an object:GetImage1, because of the limit of GPU RAM, initial image is 800x800
# Here we only obtain a 80x80 pixels 2D image
GetImage1 = GetImages(image_height=parameter.image_width, image_width=parameter.image_width)

# -------------------------------------------------------------------------------------

# Obtain dataset of train images,test images and val images
# Apply GetImage function to al the train,test,val images, modify all the images size(height, width)
train_image_dataset = (tf.data.Dataset.from_tensor_slices(train_image_path).map(GetImage1, num_parallel_calls=parameter.AUTO))
test_image_dataset = (tf.data.Dataset.from_tensor_slices(test_image_path).map(GetImage1, num_parallel_calls=parameter.AUTO))
val_image_dataset = (tf.data.Dataset.from_tensor_slices(val_image_path).map(GetImage1, num_parallel_calls=parameter.AUTO))

# -------------------------------------------------------------------------------------
# obtain focal length
focal_length = camera_focal(fov=train_jason_data["camera_angle_x"], width=parameter.image_width)

# -------------------------------------------------------------------------------------

# instantiate the GetRays object
GetRays1 = GetRays(camera_focal=focal_length, image_width=parameter.image_width, image_height=parameter.image_height,
			near_boundary=parameter.near_boundary, far_boundary=parameter.far_boundary, num_sample_coarse=parameter.num_sample_coarse)

# -------------------------------------------------------------------------------------
# dataset of rays for training ,testing, valuating
# different camera position ,from each camera position to all the image pixels (rays), num_parallel_call improve speed
train_ray_dataset = (tf.data.Dataset.from_tensor_slices(train_c2w_matrix).map(GetRays1, num_parallel_calls=parameter.AUTO))
test_ray_dataset = (tf.data.Dataset.from_tensor_slices(test_c2w_matrix).map(GetRays1, num_parallel_calls=parameter.AUTO))
val_ray_dataset = (tf.data.Dataset.from_tensor_slices(val_c2w_matrix).map(GetRays1, num_parallel_calls=parameter.AUTO))

# -------------------------------------------------------------------------------------

# combine them together each element in train_dataset is a 2-elements tuple
train_dataset = tf.data.Dataset.zip((train_ray_dataset, train_image_dataset))
test_dataset = tf.data.Dataset.zip((test_ray_dataset, test_image_dataset))
val_dataset = tf.data.Dataset.zip((val_ray_dataset, val_image_dataset))

# -------------------------------------------------------------------------------------

# Preprocess the dataset, improve training speed
train_dataset = (train_dataset.shuffle(parameter.batch_size).batch(parameter.batch_size).repeat().prefetch(parameter.AUTO))
test_dataset = (test_dataset.batch(parameter.batch_size).prefetch(parameter.AUTO))
val_dataset = (val_dataset.shuffle(parameter.batch_size).batch(parameter.batch_size).repeat().prefetch(parameter.AUTO))

# -------------------------------------------------------------------------------------

# Obtain two models, coarse model and fine model , nerf trainer model
coarse_model = MLP_model(xyz=parameter.xyz_dim, dir=parameter.dir_dim, dense_unit=parameter.dense_unit, batch_size=parameter.batch_size)
fine_model = MLP_model(xyz=parameter.xyz_dim, dir=parameter.dir_dim, dense_unit=parameter.dense_unit, batch_size=parameter.batch_size)

nerf_trainer_model = trainer(coarse_model=coarse_model, fine_model=fine_model, xyz=parameter.xyz_dim, dir=parameter.dir_dim, render_image_function=render_image_function,
encoder_function = encoder, hiera_sampling_function=hierarchical_sample, num_sample_fine=parameter.num_sample_fine)

# -------------------------------------------------------------------------------------

# Compile nerf_trainer_model, Adam optimizer is pretty good here, loss function = mean squared
nerf_trainer_model.compile(coarse_optimizer=Adam(), fine_optimizer=Adam(), loss_function=MeanSquaredError())

# -------------------------------------------------------------------------------------

# Callback: plot images function
ImagePlotCallback = plotting_epoch_img(dataset_object=test_dataset, xyz=parameter.xyz_dim, dir=parameter.dir_dim,
	encoder_function=encoder, image_save_path=parameter.image_output_path, psnr_save_path=parameter.psnr_output_path)

# -------------------------------------------------------------------------------------

nerf_trainer_model.fit(train_dataset, steps_per_epoch=parameter.step_per_epoch, validation_data=val_dataset,
				validation_steps= 5, epochs=parameter.epoch, callbacks=[ImagePlotCallback])

# -------------------------------------------------------------------------------------
# Save models, videos 3D needs these two models
nerf_trainer_model.coarse_model.save(parameter.coarse_save_path)
nerf_trainer_model.fine_model.save(parameter.fine_save_path)
