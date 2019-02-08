"""

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class PlotGanImageHook(tf.train.SessionRunHook):
    """
    Plots outputs of the generator, can be used to visualize training progress
    """

    def __init__(self, gan_model, path, every_n_iter, batch_size, name_format="gan_image_plot_{}.png"):
        self._batch_size = batch_size
        self._gan_model = gan_model
        self._path = path
        self._format = name_format
        self._every_n_iter= every_n_iter

        self._iter_count = 0

    def after_run(self, run_context, _):
        self._iter_count += 1
        if self._iter_count % self._every_n_iter != 0:
            return

        session = run_context.session
        with tf.variable_scope('Generator', reuse=True):
            # Generate images from noise, using the generator network.
            f, a = plt.subplots(1, 10, figsize=(10, 4))
            gen_output = session.run([self._gan_model.generated_data])
            gen_output = np.reshape(gen_output, (self._batch_size, 28, 28))

            # Rescale to [0, 1.0] and invert colors
            gen_output = 1.0 - (gen_output + 1.0) / 2.0

            for i in range(10):
                # Extend to 3 channels for matplot figure
                img = np.reshape(np.repeat(gen_output[i, :, np.newaxis], 3, axis=2),
                                 newshape=(28, 28, 3))
                a[i].imshow(img)

            f.show()
            plt.draw()
            plt.savefig(os.path.join(self._path, self._format.format(self._iter_count)))
            plt.close()
