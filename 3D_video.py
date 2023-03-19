"""
Most significant point : obtain all frames of the objects to form a 3D video
theta represents the camera's rotation around the y-axis of the camera coordinate,
which is the most standard horizontal rotation.
Identically, phi is respect with rotation around x-axis of camera coordinate.

"""
import tensorflow as tf
import numpy as np
from NERF3D.Nerf_data import read_json
from NERF3D.Nerf_data import GetRays
from NERF3D.Nerf_data import camera_focal
from tensorflow.keras.models import load_model
from NERF3D.Render_image import render_image_function
from NERF3D.hiera_sampling import hierarchical_sample
from NERF3D.Encoder import encoder
from NERF3D import parameter
from tqdm import tqdm
import imageio
import os

# -------------------------------------------------------------------------------------
# The radiance field describes color and volume density for every point and for every viewing direction in the scene
# F (X, θ, φ) --> (c, σ)   X --> (x, y, z)
# θ, φ represent the azimuthal and polar viewing angles

# -------------------------------------------------------------------------------------
# check if the output video directory exists, if it does not, then create it
if not os.path.exists(parameter.video_file_path):
    os.makedirs(parameter.video_file_path)

# Obtain  the translation matrix for movement in t and convert to tensor, move t units
def GetTranslation_t(t):
    t_matrix = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, t],
                [0, 0, 0, 1]]
    t_tensor = tf.convert_to_tensor(t_matrix, dtype=tf.float32)
    return t_tensor


# rotation matrix for movement in theta and convert to tensor. Horizontal
def GetRotation_theta(theta):
    theta_matrix = [[tf.cos(theta), 0, -tf.sin(theta), 0],
                   [0, 1, 0, 0],
                   [tf.sin(theta), 0, tf.cos(theta), 0],
                   [0, 0, 0, 1]]
    theta_sensor = tf.convert_to_tensor(theta_matrix, dtype=tf.float32)
    return theta_sensor


# rotation matrix for movement in phi and convert to phi. Vertical
def GetRotation_phi(phi):
    matrix = [[1, 0, 0, 0],
              [0, tf.cos(phi), -tf.sin(phi), 0],
              [0, tf.sin(phi), tf.cos(phi), 0],
              [0, 0, 0, 1]]
    phi_tensor = tf.convert_to_tensor(matrix, dtype=tf.float32)
    return phi_tensor


def point_c2w(theta, phi, t):
    """
    Compute the camera to world matrix for the corresponding theta, phi and t step by step
    By rotating the matrix get_rotation_phi(phi / 180.0 * np.pi) and multiplying it with the previous transformation matrix,
    a new transformation matrix is obtained,
    which converts the points in the camera coordinate into the world coordinate  points
    and save the result in the c2w variable
    """
    c2w = GetTranslation_t(t)
    c2w = GetRotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = GetRotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w =np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w

# define a empty frame list
frame_list = []
c2w_matrix = []
# Iterate over different theta value and generate scenes.
for theta in np.linspace(0.0, 360.0, 180, endpoint=False):
    # Get the camera to world matrix. -30 phi , 4 t are fixed, only theta with horizontal rotations
    c2w = point_c2w(theta, -30.0, 4.0)
    # add all c2w
    c2w_matrix.append(c2w)

"""
Same as train mian file's script
"""

# import train data to obtain camera focal
jsonTrainData = read_json(parameter.train_jason_path)
focalLength = camera_focal(fov=jsonTrainData["camera_angle_x"], width=parameter.image_width)

# trainImagePaths, trainC2Ws = get_imagepath_c2w(jsondata=jsonTrainData,datasetPath=config2.DATASET_PATH)

# instantiate getRays function(Getrays function input:c2w_matrix, output: ray )
getRays = GetRays(camera_focal=focalLength, image_width=parameter.image_width, image_height=parameter.image_height,
                  near_boundary=parameter.near_boundary, far_boundary=parameter.far_boundary, num_sample_coarse=parameter.num_sample_coarse)

# get the dataset of rays
RayDs = (tf.data.Dataset.from_tensor_slices(c2w_matrix).map(getRays).batch(parameter.batch_size))

# load coarse model
coarse_model = load_model(parameter.coarse_save_path, compile=False)
# load fine model
fine_model = load_model(parameter.fine_save_path, compile=False)

# -------------------------------------------------------------------------------------------
for element in tqdm(RayDs):
    (rays_ori_coarse, rays_dir_coarse, t_vals_coarse) = element
    # -------------------------------------------------------------------------------------
    coarse_ray = (rays_ori_coarse[..., None, :] + (rays_dir_coarse[..., None, :] * t_vals_coarse[..., None]))
    # -------------------------------------------------------------------------------------
    # positional encode the rays and dirs
    coarse_ray_encode =encoder(coarse_ray, parameter.xyz_dim)
    # dir shape
    coarse_ray_dir_shape = tf.shape(coarse_ray[..., :3])
    coarse_ray_dir = tf.broadcast_to(rays_dir_coarse[..., None, :], shape=coarse_ray_dir_shape)
    coarse_ray_dir_encode = encoder(coarse_ray_dir, parameter.dir_dim)

    # -------------------------------------------------------------------------------------
    """ 
    1. coarse_ray     2. coarse_dir
    2. coarse model: coarse_ray, coarse_dir ---> rgb, sigma
    3. render image function : rgb, sigma ---> image, weights(depth)

    """
    # coarse model prediction
    (coarse_rgb, coarse_sigma) =coarse_model([coarse_ray_encode, coarse_ray_dir_encode])

    # Coarse rendering
    # coarse_render is a tuple :(imagesCoarse, weightsCoarse),
    # here we only need coarse image and coarse weights
    coarse_render = render_image_function(rgb=coarse_rgb, sigma=coarse_sigma, num_sample=t_vals_coarse)
    (coarse_image, coarse_weight) = coarse_render

    # -------------------------------------------------------------------------------------

    # start applying hiera_sampling method
    # compute the middle values of t vals
    mid_t_vals_coarse = (1 / 2 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1]))
    # obtain sample will be used by fine model
    t_vals_fine1 = hierarchical_sample(mid_value=mid_t_vals_coarse, weights=coarse_weight, num_f_sample=parameter.num_sample_fine)
    # Actual t_vals_fine = coarse sample + fine sample
    t_vals_fine = tf.sort(tf.concat([t_vals_coarse, t_vals_fine1], axis=-1), axis=-1)

    # -------------------------------------------------------------------------------------
    # rays for fine model built(almost same as coarse model)
    fine_ray = (rays_ori_coarse[..., None, :] + (rays_dir_coarse[..., None, :] * t_vals_fine[..., None]))

    # -------------------------------------------------------------------------------------
    # encoding
    fine_ray_encode = encoder(fine_ray, parameter.xyz_dim)
    fine_ray_dir_shape = tf.shape(fine_ray[..., :3])
    fine_ray_dir = tf.broadcast_to(rays_dir_coarse[..., None, :], shape=fine_ray_dir_shape)
    fine_ray_dir_encode = encoder(fine_ray_dir, parameter.dir_dim)

    # -------------------------------------------------------------------------------------
    """
    1. fine_ray     2. fine_dir
    2. fine model: fine_ray, fine_dir ---> rgb, sigma
    3. render image function : rgb, sigma ---> image, weights(depth)

    """
    # compute the predictions from the fine model
    fine_rgb, fine_sigma = fine_model([fine_ray_encode, fine_ray_dir_encode])

    # fine_render is a tuple :(images fine, weights fine),
    # here only need fine images
    fine_render = render_image_function(rgb=fine_rgb, sigma=fine_sigma, num_sample=t_vals_fine)
    (fine_image, _) = fine_render

    # insert the rendered fine image to the collection
    frame_list.append(fine_image.numpy()[0])


# output video final result
imageio.mimwrite(parameter.video_output_path, frame_list, fps=parameter.fps, quality=parameter.video_quality)





