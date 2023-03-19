"""
OPTIMIZATION: Hierarchical volume sampling
"""
import tensorflow as tf
from .parameter import batch_size
from .parameter import image_width
from .parameter import image_height

import numpy as np
"""
mid_value: mid of value between adjacent sample points
weight from image render
num_f_sample : number of samples used by fine model
c formular
"""


def hierarchical_sample(mid_value, weights, num_f_sample):
    # add a small value to the weights to prevent it is zero
    weights = weights + 1e-5
    # normalize the weights to get a probability density function, formular
    pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # Obtain cdf(cumulative distribution function)
    # Cumulative addition along with the last dimension, tf.cumsum
    cdf = tf.cumsum(pdf, axis=-1)
    # Make sure the element in left boundary is 0, it is useful for calculation
    padding = tf.zeros_like(cdf[..., :1])
    cdf = tf.concat([padding, cdf], axis=-1)

    # ---------------------------------------------------------------------
    # Random uniform sampling,shape= point_shape
    point_shape= [batch_size, image_height, image_width, num_f_sample]
    u = tf.random.uniform(shape=point_shape)

    # ---------------------------------------------------------------------
    # obtain index of u, side = right finds an index that bigger than u in cdf
    index = tf.searchsorted(cdf, u, side="right")
    # Get the max and min boundaries, samples in boundary are what we prefer
    # index - 1 let side
    min = tf.maximum(0, index - 1)

    # cdf.shape[-1] - 1 ------ cdf right side index
    max = tf.minimum(cdf.shape[-1] - 1, index)
    indices = tf.stack([min, max], axis=-1)

    # ---------------------------------------------------------------------
    cdf1 = tf.gather(cdf, indices, axis=-1, batch_dims=len(indices.shape) - 2)
    mid_tvalue1 = tf.gather(mid_value, indices, axis=-1, batch_dims=len(indices.shape) - 2)

    # Inverse transform sampling method, first obtain normalized denominate
    denominate = cdf1[..., 1] - cdf1[..., 0]
    denominate = tf.where(denominate < 1e-5, tf.ones_like(denominate), denominate)
    t = (u - cdf1[..., 0]) / denominate
    model_samples = mid_tvalue1[..., 0] + t * (mid_tvalue1[..., 1] - mid_tvalue1[..., 0])

    return model_samples
