"""
# Achieve rendering , according to the rendering formular
# Inputs: rgb ,sigma
# new views of same scene
# -----------------------------------------------
"""

from .parameter import batch_size
from .parameter import image_width
from .parameter import image_height
import tensorflow as tf

# sigma = density in specific point
# rgb and sigma are obtained from MLP model (coarse or fine)
def render_image_function(rgb, sigma, num_sample):
    # squeeze the last dimension of sigma
    sigma = sigma[..., 0]
    # -------------------------------------------------------------------------
    # Calculate the difference between adjacent dimensions for each element
    # δi delta is the distance from sample i to sample i + 1.
    delta = num_sample[..., 1:] - num_sample[..., :-1]
    # Define delta shape
    delta_shape = [batch_size, image_height, image_width, 1]
    # Add a new dimension in delta, all the dimensional elements are very big value (infinite) make sure
    # all the points will be measured
    delta = tf.concat([delta, tf.broadcast_to([1e10], shape=delta_shape)], axis=-1)

    # -------------------------------------------------------------------------
    """
     accumulative (alpha_i * transmittance * rgb)
     output = color matrix + weight of each sample point
     Critically based on the rendering formula
     
    """
    bias_avoid_zero = 1e-10
    # light Attenuation at sample point i
    alpha_i = 1.0 - tf.exp(-sigma * delta)
    # How much light through between sample i to sample i+1 ray/ transparency = 1 - alpha_i
    transparency_light = 1.0 - alpha_i
    # -------------------------------------------------------------------------

    # aims to obtain transmittance of the light along the ray
    # calculate the transmittance and weights of the ray points
    transmittance = tf.math.cumprod(transparency_light + bias_avoid_zero, axis=-1, exclusive=True)
    # weight
    weights = alpha_i * transmittance

    """
    # rgb : (batch size, image height, image width, num_sample,3)
    # weights:   (batch size, image height, image width, num_sample)
    
    """
    # color matrix (height, width, 3)
    image = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

    # return image,weights
    return image, weights
