"""
Contain read_json function, camera focal function,get image path and each corresponding c2w matrix function,
Get images function and Get rays function (ray formular)
import tensorflow as tf to build the data pipline
Create input data pipe for the image
import json: in order to deal with json file,which contains the frames and focal length
"""

import tensorflow as tf
import json
from tensorflow import reshape

# -----------------------------------------------------------------------------------
# define read _json file function/ use for load json file and return the data
def read_json(jsonPath):
    # open the json file
    with open(jsonPath, "r") as fp:
        # read the json data
        data = json.load(fp)
    # return the data
    return data

# -----------------------------------------------------------------------------------
# camer_focal function: obtain camera focal length：f from the field of view
def camera_focal(fov, width):
    # Initial formular: fov = 2 arctan(width/2f)
    # simplify equation
    f = (width/2)/tf.tan(fov/2)
    return f

# -----------------------------------------------------------------------------------
# load images from the dataset
def get_imagepath_c2w(jsondata, datasetPath):
    # define a list to store the image paths
    imagepaths = []
    # list for store the camera to world matrices
    matraix_c2w = []

    # iterate over each frame of the data
    for frame in jsondata["frames"]:
        # grab the image file name, here frame is also a list which contains file_path, rotation ,transform_matrix
        imagepath = frame["file_path"]
        # all the images are located dataset, initial file_path are './train/r_x' x is the order of graph
        # replace '.' by datasetpath
        imagepath = imagepath.replace(".", datasetPath)
        # add elements to new list imagepaths
        imagepaths.append(f"{imagepath}.png")
        # grab the camera to world matrix, each picture has its own specific camera to world transform_matrix
        matraix_c2w.append(frame["transform_matrix"])

    # return image paths and camera to world transform matrices
    return imagepaths, matraix_c2w

# -----------------------------------------------------------------------------------


class GetImages:
    # Obtain actually image for us
    # initialize parameters
    def __init__(self, image_width, image_height):
        # define the image width and height
        self.image_width = image_width
        self.image_height = image_height

    def __call__(self, imagePath):
        # Get initial image
        image = tf.io.read_file(imagePath)

        # Decode image string in a jpeg format,output RGB graph
        image = tf.image.decode_jpeg(image, 3)

        # convert the image dtype from uint8 to float32, float 32 can improve rendering quality
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Give the image height and image width in config file
        image = tf.image.resize(image, (self.image_width, self.image_height))
        # Add color chanels, if directly use resize add color chanel will make image distortion
        image = reshape(image, (self.image_width, self.image_height, 3))
        return image
# -----------------------------------------------------------------------------------


class GetRays:
    def __init__(self, camera_focal, image_width, image_height, near_boundary, far_boundary, num_sample_coarse):

        self.camera_focal = camera_focal
        self.image_width = image_width
        self.image_height = image_height
        # Two camera viewing fields
        self.near_boundary = near_boundary
        self.far_boundary = far_boundary
        self.num_sample_coarse = num_sample_coarse

    # start creating the rays
    def __call__(self, matrix_c2w):
        # Creates meshgrid. A 2D grid generates ray directions from the camera position
        (x, y) = tf.meshgrid(
            tf.range(self.image_width, dtype=tf.float32),
            tf.range(self.image_height, dtype=tf.float32), indexing="xy",)

        # camera coordinates, according to the equation between camera and image plane
        # xc = zc*(U-Ox)/f , yc = zc*(V-Oy)/f , (Ox, Oy) is the centre of image plane
        camera_x = (x - self.image_width * 1/2) / self.camera_focal
        camera_y = (y - self.image_height * 1/2) / self.camera_focal

        # define the camera vector, here -tf.one_like set all zC to minus one
        # Stack or merge operations along the column dimension, in order to multiply 3x3 rotation
        # The direction of the image y-axis is downward, which is opposite to the direction of the camera y-axis
        xcyczc = tf.stack([camera_x, -camera_y, -tf.ones_like(x)], axis=-1)
        xcyczc = xcyczc[..., None, :]

        # get the rotation matrix and translation matrix  from the camera-to-world matrix
        # 3X3 rotation matrix
        rotation = matrix_c2w[:3, :3]
        # 3X1 translation matrix, as well as start point of the ray, initial point of the camera coordinate
        translation = matrix_c2w[:3, -1]

        # world coordinates
        xwywzw = xcyczc * rotation

        # ----------------------------------------------------------------------------
        # calculate the direction vector of the ray
        """
        Ray_ori original point
        Ray_dir set of direction vector
        sampled_p sampled points
        """

        #  direction vector of the ray, vector calculation method
        ray_dir = tf.reduce_sum(xwywzw, axis=-1)
        # use tf.norm to calculate in the last dimension , unit direction vector of the ray
        ray_dir = ray_dir / tf.norm(ray_dir, axis=-1, keepdims=True)

        # tf.broadcast_to multi translation matrix ,calculate the origin vector of the ray
        # translation matrix is the location of camera as well as ray original point shape is same as ray_dir
        ray_ori = tf.broadcast_to(translation, tf.shape(ray_dir))

        # ----------------------------------------------------------------------------
        # get some sample points on the ray, sample uniformly number of sample_num points
        sample_p = tf.linspace(self.near_boundary, self.far_boundary, self.num_sample_coarse)
        noiseShape = list(ray_ori.shape[:-1]) + [self.num_sample_coarse]

        # use tf.random.uniform Outputs random values from  shape = noiseshape, uniform
        """
        Adds noise for color and orientation offsets at each sample point.
        Improve image quality and make rendering results more realistic
        """
        noise = (tf.random.uniform(shape=noiseShape) * (self.far_boundary - self.near_boundary) / self.num_sample_coarse)
        sample_p = sample_p + noise
        # ray formula : r(t) = o + td
        # respective to world coordinate
        return (ray_ori, ray_dir, sample_p)
