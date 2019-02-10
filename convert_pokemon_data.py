# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import os
import imageio

import numpy as np
import tensorflow as tf

from tf_gan_research_deps import dataset_utils

_IMAGE_SIZE = 32
_NUM_CHANNELS = 4
_MAX_THREADS = 4
_CLASS_NAMES = ["fake", "real"]


def _extract_images(image_paths):
    """Extract the images into a numpy array.

    Args:
      filename: The path to an MNIST images file.
      num_images: The number of images in the file.

    Returns:
      A numpy array of shape [number_of_images, height, width, channels].
    """

    num_images = len(image_paths)
    data = np.zeros((num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS))
    for i in range(num_images):
        image_path = image_paths[i]
        print('Extracting images from: ', image_path)
        image = imageio.imread(image_path)
        data[i] = image

    return data


def _load_image_paths(image_dir):
    return [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]


def _load_images_and_labels(image_dir):
    """Extract the images into a numpy array.

    Args:
      image_dir: The path to a Pokemon images file.

    Returns:
      A tuple with a numpy array containing the images, the labels, and the number of images
    """

    print('Extracting images from: ', image_dir)

    image_paths = _load_image_paths(image_dir)
    images = _extract_images(image_paths)
    num_images = len(image_paths)
    labels = np.ones(num_images, dtype=np.int64)

    return images, labels


def _add_to_tfrecord(images, labels, tfrecord_writer):
    """Loads data from the binary Pokemon PNG and writes files to a TFRecord.

    Args:
      image_dir: The directory containing the Pokemon images.
      tfrecord_writer: The TFRecord writer to use for writing.
    """

    shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)

        with tf.Session('') as sess:
            num_images = len(images)
            for i in range(num_images):
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, num_images))
                sys.stdout.flush()

                png_string = sess.run(encoded_png, feed_dict={image: images[i]})

                example = dataset_utils.image_to_tfexample(
                    png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[i])
                tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir):
    """Creates the output filename.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return os.path.join(dataset_dir, 'pokemon.tfrecord')


def run(input_dir, dataset_dir):
    """Runs the conversion operation.

    Args:
      input_dir: The input directory
      dataset_dir: The dataset directory where the dataset is stored.
    """

    if not tf.gfile.Exists(input_dir):
        print('Input dir is empty. Exiting...')
        return

    tfrecord_filename = _get_output_filename(dataset_dir)
    if tf.gfile.Exists(tfrecord_filename):
        print('Dataset files already exist. Delete the files- if you want to reprocess. Exiting...')
        return

    tf.gfile.MakeDirs(dataset_dir)
    with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
        images, labels = _load_images_and_labels(input_dir)
        _add_to_tfrecord(images, labels, tfrecord_writer)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    print('\nFinished converting the Pokemon dataset!')


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_dir',
    None,
    'The directory where the Pokemon PNG images are stored')


tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')


def main(_):
    if not FLAGS.input_dir:
        raise ValueError('You must supply the input directory with --input_dir')

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    run(FLAGS.input_dir, FLAGS.dataset_dir)


if __name__ == '__main__':
    tf.app.run()
