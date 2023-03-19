# import necessary packages
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.metrics import Mean


class trainer(tf.keras.Model):
    """
    Training process: coarse model ---> fine model
    xyz : the number of dimensions used for positional encoding of the xyz coordinates
    dir: the number of dimensions used for positional encoding of the direction vector
    coarse model + fine model produce higher quality of rendering image depth function
    encoder function
    num_sample_fine : the number of samples that fine model going to take
    """

    # -------------------------------------------------------------------------------------
    def __init__(self, coarse_model, fine_model, xyz, dir, render_image_function, encoder_function, hiera_sampling_function, num_sample_fine):
        super().__init__()
        # define the coarse model and fine model
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.xyz = xyz
        self.dir = dir
        self.encoder_function = encoder_function
        self.render_image_function = render_image_function
        self.hiera_sampling_function = hiera_sampling_function
        self.num_sample_fine = num_sample_fine

    # -------------------------------------------------------------------------------------
    def compile(self, coarse_optimizer, fine_optimizer, loss_function):
        super().compile()
        # initialize all parameters
        self.coarse_optimizer = coarse_optimizer
        self.fine_optimizer = fine_optimizer
        # loss_func : photometric loss function
        self.loss_function = loss_function
        # Track loss, Mean function , gives Mean function's name
        self.loss_tracker = Mean(name="loss")
        # Peak Signal - to - Noise Ratio(PSNR)
        self.psnr_tracker = Mean(name="psnr")

    # -------------------------------------------------------------------------------------
    def train_step(self, inputs):
        (rays_elements, images) = inputs
        # for single coarse model
        (rays_ori_coarse, rays_dir_coarse, t_vals_coarse) = rays_elements

        """
        ray_ori_coarse : starting point of num_coarse_samples rays in the space after camera-to-world 
        coordinate transformation.  

        ray_dir_coarse : direction vector for each ray

        t_vals_coarse :  num_coarse_samples points 
        sampled along the direction vector on each ray to the starting point of the ray./depth

        """
        # tf.newaxis: increase one more dimension
        # -------------------------------------------------------------------------------------
        # rays for coarse model built ,  r(t) = o + td
        # coarse_ray:(batch_size, num_coarse_samples, 3)
        # rays_ori_coarse:(batch_size,num_rays, 3)---->(batch_size,num_rays,1, 3)
        # rays_dir_coarse:(batch_size,num_rays, 3)---->(batch_size,num_rays,1, 3)
        # t_vals_coarse :(batch_size,num_rays, num_coarse_sample) ---->(batch_size, num_rays, num_coarse_sample, 1)

        coarse_ray = (rays_ori_coarse[..., None, :] + (rays_dir_coarse[..., None, :] * t_vals_coarse[..., None]))
        # -------------------------------------------------------------------------------------

        # positional encode the rays and dirs
        coarse_ray_encode = self.encoder_function(coarse_ray, self.xyz)
        # dir shape, get first 3 element in last dimension means direction v
        coarse_ray_dir_shape = tf.shape(coarse_ray[..., :3])
        # coarse_ray_dir : (batch_size, num_rays, num_coarse_sample, 3)
        coarse_ray_dir = tf.broadcast_to(rays_dir_coarse[..., None, :], shape=coarse_ray_dir_shape)
        coarse_ray_dir_encode = self.encoder_function(coarse_ray_dir, self.dir)

        # -------------------------------------------------------------------------------------
        """ 
        1. coarse_ray     2. coarse_dir
        2. coarse model: coarse_ray, coarse_dir ---> rgb, sigma
        3. render image function : rgb, sigma ---> image, weights(depth)

        """
        with tf.GradientTape() as coarse_tape:
            # coarse model prediction,
            # coarse_ray_encode ---> the xyx part of 5D coordinate(x, y, z, θ, φ)
            # coarse_ray_dir_encode ---> θ,φ part of 5D coordinate
            (coarse_rgb, coarse_sigma) = self.coarse_model([coarse_ray_encode, coarse_ray_dir_encode])

            # Coarse rendering
            # coarse_render is a tuple :(imagesCoarse, weightsCoarse),
            coarse_render = self.render_image_function(rgb=coarse_rgb, sigma=coarse_sigma, num_sample=t_vals_coarse)
            (coarse_image, coarse_weight) = coarse_render

            # Calculate loss
            coarse_loss = self.loss_function(images, coarse_image)
        # -------------------------------------------------------------------------------------

        # start applying hiera_sampling method
        # compute the middle values of t vals
        mid_t_vals_coarse = (1/2 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1]))
        # obtain sample will be used by fine model
        t_vals_fine1 = self.hiera_sampling_function(mid_value=mid_t_vals_coarse, weights=coarse_weight, num_f_sample=self.num_sample_fine)
        # Actual t_vals_fine = coarse sample + fine sample
        t_vals_fine = tf.sort(tf.concat([t_vals_coarse, t_vals_fine1], axis=-1), axis=-1)

        # -------------------------------------------------------------------------------------
        # rays for fine model built(almost same as coarse model)
        fine_ray = (rays_ori_coarse[..., None, :] + (rays_dir_coarse[..., None, :] * t_vals_fine[..., None]))

        # -------------------------------------------------------------------------------------
        # encoding
        fine_ray_encode = self.encoder_function(fine_ray, self.xyz)
        fine_ray_dir_shape = tf.shape(fine_ray[..., :3])
        fine_ray_dir = tf.broadcast_to(rays_dir_coarse[..., None, :], shape=fine_ray_dir_shape)
        fine_ray_dir_encode = self.encoder_function(fine_ray_dir, self.dir)

        # -------------------------------------------------------------------------------------
        """
        1. fine_ray     2. fine_dir
        2. fine model: fine_ray, fine_dir ---> rgb, sigma
        3. render image function : rgb, sigma ---> image, weights(depth)
        
        """
        with tf.GradientTape() as fine_tape:
            # compute the predictions from the fine model
            fine_rgb, fine_sigma = self.fine_model([fine_ray_encode, fine_ray_dir_encode])

            # fine_render is a tuple has images fine, weights fine
            # here only need images
            fine_render = self.render_image_function(rgb=fine_rgb, sigma=fine_sigma, num_sample=t_vals_fine)
            (fine_image, _) = fine_render

            # Calculate loss
            fine_loss = self.loss_function(images, fine_image)
        # -------------------------------------------------------------------------------------
        # backpropagation

        # coarse model optimizer update, trainable value weight
        coarse_trainable_val = self.coarse_model.trainable_variables

        # calculate gradient，
        coarse_gradient = coarse_tape.gradient(coarse_loss, coarse_trainable_val)
        # apply gradient to , zip --> (gradient,coarse_variable)
        self.coarse_optimizer.apply_gradients(zip(coarse_gradient, coarse_trainable_val))

        # fine model optimizer update
        fine_trainable_val = self.fine_model.trainable_variables
        fine_gradient = fine_tape.gradient(fine_loss, fine_trainable_val)
        self.fine_optimizer.apply_gradients(zip(fine_gradient, fine_trainable_val))

        #  have to set max_val, otherwise error occurs
        psnr = tf.image.psnr(images, fine_image, max_val=1.0)
        # -------------------------------------------------------------------------------------
        # update
        self.loss_tracker.update_state(fine_loss)
        self.psnr_tracker.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_tracker.result()}

        # -------------------------------------------------------------------------------------

    # test step and train step are identical, but test step does not have gradients
    def test_step(self, inputs):
        (rays_elements, images) = inputs
        # for single coarse model
        # -------------------------------------------------------------------------------------
        (rays_ori_coarse, rays_dir_coarse, t_vals_coarse) = rays_elements
        # -------------------------------------------------------------------------------------
        coarse_ray = (rays_ori_coarse[..., None, :] + (rays_dir_coarse[..., None, :] * t_vals_coarse[..., None]))
        # -------------------------------------------------------------------------------------
        # positional encode the rays and dirs
        coarse_ray_encode = self.encoder_function(coarse_ray, self.xyz)
        # dir shape
        coarse_ray_dir_shape = tf.shape(coarse_ray[..., :3])
        coarse_ray_dir = tf.broadcast_to(rays_dir_coarse[..., None, :], shape=coarse_ray_dir_shape)
        coarse_ray_dir_encode = self.encoder_function(coarse_ray_dir, self.dir)

        # -------------------------------------------------------------------------------------
        """ 
        1. coarse_ray     2. coarse_dir
        2. coarse model: coarse_ray, coarse_dir ---> rgb, sigma
        3. render image function : rgb, sigma ---> image, weights(depth)

        """
        # coarse model prediction
        (coarse_rgb, coarse_sigma) = self.coarse_model([coarse_ray_encode, coarse_ray_dir_encode])

        # Coarse rendering
        # fine_render is a tuple has images fine, weights fine
        coarse_render = self.render_image_function(rgb=coarse_rgb, sigma=coarse_sigma, num_sample=t_vals_coarse)
        (coarse_image, coarse_weight) = coarse_render

        # -------------------------------------------------------------------------------------

        # start applying hiera_sampling method
        # compute the middle values of t vals
        mid_t_vals_coarse = (1 / 2 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1]))
        # obtain sample will be used by fine model
        t_vals_fine1 = self.hiera_sampling_function(mid_value=mid_t_vals_coarse, weights=coarse_weight, num_f_sample=self.num_sample_fine)
        # Actual t_vals_fine = coarse sample + fine sample
        t_vals_fine = tf.sort(tf.concat([t_vals_coarse, t_vals_fine1], axis=-1), axis=-1)

        # -------------------------------------------------------------------------------------
        # rays for fine model built(almost same as coarse model)
        fine_ray = (rays_ori_coarse[..., None, :] + (rays_dir_coarse[..., None, :] * t_vals_fine[..., None]))

        # -------------------------------------------------------------------------------------
        # encoding
        fine_ray_encode = self.encoder_function(fine_ray, self.xyz)
        fine_ray_dir_shape = tf.shape(fine_ray[..., :3])
        fine_ray_dir = tf.broadcast_to(rays_dir_coarse[..., None, :], shape=fine_ray_dir_shape)
        fine_ray_dir_encode = self.encoder_function(fine_ray_dir, self.dir)

        # -------------------------------------------------------------------------------------
        """
        1. fine_ray     2. fine_dir
        2. fine model: fine_ray, fine_dir ---> rgb, sigma
        3. render image function : rgb, sigma ---> image, weights(depth)

        """
        # compute the predictions from the fine model
        fine_rgb, fine_sigma = self.fine_model([fine_ray_encode, fine_ray_dir_encode])

        # fine_render is a tuple has images fine, weights fine
        # here only need fine images
        fine_render = self.render_image_function(rgb=fine_rgb, sigma=fine_sigma, num_sample=t_vals_fine)
        (fine_image, _) = fine_render

        # -------------------------------------------------------------------------------------
        # Calculate loss and PSNR metrics
        fine_loss = self.loss_function(images, fine_image)
        #  have to set max_val, otherwise error occurs
        psnr = tf.image.psnr(images, fine_image, max_val=1.0)

        # -------------------------------------------------------------------------------------
        # update
        self.loss_tracker.update_state(fine_loss)
        self.psnr_tracker.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_tracker.result()}

        # -------------------------------------------------------------------------------------

    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_tracker]

    # ------------------------------------------------------------------------------------------
"""
    
Aim to make the whole training process more clear
Here I need print out coarse model image and fine model image after each single epoch and loss image
Method: use callback function, define corresponding plotting method in (on_epoch_end)
    
"""


def plotting_epoch_img(dataset_object, xyz, dir, encoder_function, image_save_path, psnr_save_path):

    (rays_element, images) = next(iter(dataset_object))
    # for single coarse model
    (rays_ori_coarse, rays_dir_coarse, t_vals_coarse) = rays_element
    # -------------------------------------------------------------------------------------
    # rays for coarse model built
    coarse_ray = (rays_ori_coarse[..., None, :] + (rays_dir_coarse[..., None, :] * t_vals_coarse[..., None]))
    # -------------------------------------------------------------------------------------

    # positional encode the rays and dirs(for coarse ray)
    coarse_ray_encode = encoder_function(coarse_ray, xyz)
    # dir shape
    coarse_ray_dir_shape = tf.shape(coarse_ray[..., :3])
    coarse_ray_dir = tf.broadcast_to(rays_dir_coarse[..., None, :], shape=coarse_ray_dir_shape)
    coarse_ray_dir_encode = encoder_function(coarse_ray_dir, dir)

    # -------------------------------------------------------------------------------------
    psnr_list = []
    loss_list = []
    class PlottingImage(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # loss image
            lossa = logs["loss"]
            loss_list.append(lossa)
            a = logs["psnr"]
            psnr_list.append(a)
            # -------------------------------------------------------------------------------------
            # starting getting coarse model prediction images & fine model prediction images
            # coarse model prediction
            (coarse_rgb, coarse_sigma) = self.model.coarse_model.predict([coarse_ray_encode, coarse_ray_dir_encode])

            # Coarse rendering
            # coarse_render is a tuple :(imagesCoarse, weightsCoarse),
            # here we only need coarse image and coarse weights
            coarse_render = self.model.render_image_function(rgb=coarse_rgb, sigma=coarse_sigma, num_sample=t_vals_coarse)
            (coarse_image, coarse_weight) = coarse_render

            # -------------------------------------------------------------------------------------

            # start applying hiera_sampling method
            # compute the middle values of t vals
            mid_t_vals_coarse = (1 / 2 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1]))
            # obtain sample will be used by fine model
            t_vals_fine1 = self.model.hiera_sampling_function(mid_value=mid_t_vals_coarse, weights=coarse_weight, num_f_sample=self.model.num_sample_fine)
            # Actual t_vals_fine = coarse sample + fine sample
            t_vals_fine = tf.sort(tf.concat([t_vals_coarse, t_vals_fine1], axis=-1), axis=-1)

            # -------------------------------------------------------------------------------------
            # rays for fine model built(almost same as coarse model)
            fine_ray = (rays_ori_coarse[..., None, :] + (rays_dir_coarse[..., None, :] * t_vals_fine[..., None]))

            # -------------------------------------------------------------------------------------
            # encoding(fine ray)
            fine_ray_encode = self.model.encoder_function(fine_ray, xyz)
            fine_ray_dir_shape = tf.shape(fine_ray[..., :3])
            fine_ray_dir = tf.broadcast_to(rays_dir_coarse[..., None, :], shape=fine_ray_dir_shape)
            fine_ray_dir_encode = self.model.encoder_function(fine_ray_dir, dir)

            # -------------------------------------------------------------------------------------

            # compute the predictions from the fine model
            fine_rgb, fine_sigma = self.model.fine_model.predict([fine_ray_encode, fine_ray_dir_encode])

            # fine_render is a tuple :(images fine, weights fine),
            # here only need fine images
            fine_render = self.model.render_image_function(rgb=fine_rgb, sigma=fine_sigma, num_sample=t_vals_fine)
            (fine_image, _) = fine_render
            # fine image :(batch size, H, W, 4)
            # -------------------------------------------------------------------------------------

            """
            plot
            In the nerf, The image obtained  after rendering is saved in the form of an array (numpy array)
            """

            (fig, ax) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
            # add a main title
            fig.suptitle('3D image reconstruction(hotdog)', fontsize=22)
            # show the coarse image
            # inverse y-axis , 0-80
            ax[0, 0].invert_yaxis()
            ax[0, 0].imshow(array_to_img(coarse_image[0]), extent=(0, 80, 0, 80))
            # set title
            title0 = ax[0, 0].set_title(f"Coarse model image: {epoch:03d}", fontsize=15)
            # set color
            title0.set_color('red')
            # -------------------------------------------------------------------------------------

            ax[0, 1].invert_yaxis()
            ax[0, 1].imshow(array_to_img(fine_image[0]), extent=(0, 80, 0, 80))
            title1 = ax[0, 1].set_title(f"Fine model image: {epoch:03d}", fontsize=15)
            title1.set_color('red')

            # -------------------------------------------------------------------------------------
            ax[1, 0].invert_yaxis()
            ax[1, 0].imshow(array_to_img(images[0]), extent=(0, 80, 0, 80))
            title2 = ax[1, 0].set_title(f"Real image: {epoch:03d}", fontsize=15)
            title2.set_color('red')

            ax[1, 1].plot(loss_list, color='red')
            ax[1, 1].grid(True)
            ax[1, 1].set_xlabel('Epoch')
            ax[1, 1].set_ylabel('Loss')
            ax[1, 1].set_xticks(np.arange(0, 201, 25))  # 10.0
            title3 = ax[1, 1].set_title(f"Loss: {epoch:03d}", fontsize=15)
            title3.set_color('red')
            plt.savefig(f"{image_save_path}/{epoch:03d}.png")
            # -------------------------------------------------------------------------------------
            # psnr image
            plt.figure()
            plt.plot(psnr_list, color='red')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('PSNR')
            plt.xticks(np.arange(0, 201, 25))  # 10.0
            title4 = plt.title(f"PSNR: {epoch:03d}", fontsize=15)
            title4.set_color('red')

            plt.savefig(f"{psnr_save_path}/{epoch:03d}.png")
            plt.close('all')

    # Instantiation
    ImagePlot = PlottingImage()
    # return the train monitor
    return ImagePlot










