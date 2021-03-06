"""

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""
import math

import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class PlotGanImageHook(tf.train.SessionRunHook):
    """
    Plots outputs of the generator, can be used to visualize training progress
    """

    def __init__(self, gan_model, path, every_n_iter, batch_size, image_size,
                 name_format="gan_image_plot_{}.png"):
        self._image_size = image_size
        self._batch_size = batch_size
        self._gan_model = gan_model
        self._path = path
        self._format = name_format
        self._every_n_iter = every_n_iter

        self._iter_count = 0
        if not tf.gfile.Exists(self._path):
            tf.gfile.MakeDirs(self._path)

    def after_run(self, run_context, _):
        self._iter_count += 1
        if self._iter_count % self._every_n_iter != 0:
            return

        session = run_context.session
        with tf.variable_scope('Generator', reuse=True):
            # Generate images from noise, using the generator network.
            gen_output = session.run([self._gan_model.generated_data])
            gen_output = np.reshape(gen_output,
                                    (self._batch_size, self._image_size[0], self._image_size[1], -1))

            # Rescale to [0, 1.0] and invert colors
            gen_output = 1.0 - (gen_output + 1.0) / 2.0

            grid_side = math.ceil(math.sqrt(self._batch_size))
            f, a = plt.subplots(grid_side, grid_side, figsize=(20, 20))
            for i in range(grid_side):
                for j in range(grid_side):
                    img_idx = i * grid_side + j
                    if img_idx >= self._batch_size:
                        break

                    img = gen_output[img_idx]
                    if self._image_size[2] == 1:
                        # Extend grayscale to 3 channels for matplot figure
                        img = np.reshape(np.repeat(img, 3, axis=2), newshape=(self._image_size[0], self._image_size[1], 3))
                    elif self._image_size[2] == 4:
                        # Remove transparency layer in case of RGBA
                        img = img[:, :, 0:3]
                    a[i][j].imshow(img)

            f.show()
            plt.draw()
            plt.savefig(os.path.join(self._path, self._format.format(self._iter_count)))
            plt.close()
