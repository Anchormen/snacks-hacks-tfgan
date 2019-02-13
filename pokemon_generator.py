from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging
from plot_gan_image_hook import PlotGanImageHook
import tensorflow as tf

tfgan = tf.contrib.gan
layers = tf.contrib.layers
slim = tf.contrib.slim

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_string('train_log_dir', '/tmp/pokemon/',
                    'Directory where to write event logs.')

flags.DEFINE_string('dataset_dir', '/tmp/pokemon_data', 'Directory where the TFRecord data resides')

flags.DEFINE_string('tfr_filename', 'pokemon.tfrecord', 'TFRecord filename')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'grid_size', 5, 'Grid size for image visualization.')

flags.DEFINE_integer(
    'noise_dims', 64, 'Dimensions of the generator noise vector.')

flags.DEFINE_float(
    'generator_learning_rate', 1e-3, 'Learning rate of the generator')

flags.DEFINE_float(
    'discriminator_learning_rate', 1e-4, 'Learning rate of the discriminator')

flags.DEFINE_integer('image_size', 32, 'Image side length of the square image')
flags.DEFINE_integer('num_channels', 4, 'Image side length of the square image')

FLAGS = flags.FLAGS


def _leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)


def unconditional_generator(noise, weight_decay=2.5e-5, is_training=True):
    """Generator to produce unconditional Pokemon images. Inspired by the generator in tensorflow.models.research.gan.mnist.networks

    Args:
      noise: A single Tensor representing noise.
      weight_decay: The value of the l2 weight decay.
      is_training: If `True`, batch norm uses batch statistics. If `False`, batch
        norm uses the exponential moving average collected from population
        statistics.

    Returns:
      A generated image in the range [-1, 1].
    """

    with tf.contrib.framework.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)):
        with tf.contrib.framework.arg_scope(
                [layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(noise, 1024)
            net = layers.fully_connected(net, 1024)
            net = layers.fully_connected(net, 8 * 8 * 128)
            net = tf.reshape(net, [-1, 8, 8, 128])
            net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
            net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
            # ie [-1, 1].
            net = layers.conv2d(
                net, FLAGS.num_channels, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

            return net


def unconditional_discriminator(img, weight_decay=2.5e-5):
    """Discriminator network on unconditional Pokemon digits. Inspired by the generator in tensorflow.models.research.gan.mnist.networks

    Args:
      img: Real or generated Pokemon. Should be in the range [-1, 1].
      weight_decay: The L2 weight decay.

    Returns:
      Logits for the probability that the image is real.
    """
    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=_leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)

    return layers.linear(net, 1)


def _read_label_file(dataset_dir, filename="labels.txt"):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names


def _load_dataset(dataset_dir, file_name, num_readers=2, num_threads=2):
    """Gets a dataset tuple with instructions for reading Pokemon data.

    Args:
      dataset_dir: The base directory of the dataset sources.
      file_name:
      num_readers:
      num_threads:

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """

    file_pattern = os.path.join(dataset_dir, file_name)
    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        'image/class/label': tf.FixedLenFeature(
            [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels],
                                              channels=FLAGS.num_channels),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = _read_label_file(dataset_dir)

    dataset = slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=None,
        num_classes=len(labels_to_names),
        items_to_descriptions=None,
        labels_to_names=labels_to_names)

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size,
        shuffle=True)
    [image, label] = provider.get(['image', 'label'])

    # Creates a QueueRunner for the pre-fetching operation.
    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=num_threads,
        capacity=5 * FLAGS.batch_size)

    one_hot_labels = tf.one_hot(labels, dataset.num_classes)
    return images, one_hot_labels, dataset.num_samples


def _rescale_images(images):
    return (tf.to_float(images) - 128.0) / 128.0


def main(_):
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.name_scope('inputs'):
        with tf.device('/cpu:0'):
            images, one_hot_labels, _ = _load_dataset(FLAGS.dataset_dir, FLAGS.tfr_filename, num_readers=2,
                                                      num_threads=2)
            images = _rescale_images(images)

    generator_fn = unconditional_generator
    noise_fn = tf.random_normal(
        [FLAGS.batch_size, FLAGS.noise_dims])
    gan_model = tfgan.gan_model(
        generator_fn=generator_fn,
        discriminator_fn=unconditional_discriminator,
        real_data=images,
        generator_inputs=noise_fn)

    tfgan.eval.add_gan_model_image_summaries(gan_model, FLAGS.grid_size)

    # Get the GANLoss tuple. You can pass a custom function, use one of the
    # already-implemented losses from the losses library, or use the defaults.
    with tf.name_scope('loss'):

        gan_loss = tfgan.gan_loss(
            gan_model,
            gradient_penalty_weight=1.0,
            mutual_information_penalty_weight=0.0,
            add_summaries=True)
        # tfgan.eval.add_regularization_loss_summaries(gan_model)

    # Get the GANTrain ops using custom optimizers.
    with tf.name_scope('train'):
        train_ops = tfgan.gan_train_ops(
            gan_model,
            gan_loss,
            generator_optimizer=tf.train.AdamOptimizer(FLAGS.generator_learning_rate, 0.5),
            discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.discriminator_learning_rate, 0.5),
            summarize_gradients=True,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    # Run the alternating training loop. Skip it if no steps should be taken
    # (used for graph construction tests).
    status_message = tf.string_join(
        ['Starting train step: ',
         tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')
    if FLAGS.max_number_of_steps == 0:
        return

    gan_plotter_hook = PlotGanImageHook(gan_model=gan_model, path=os.path.join(os.sep, "tmp", "gan_output"),
                                        every_n_iter=500, batch_size=FLAGS.batch_size,
                                        image_size=(FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))

    tfgan.gan_train(
        train_ops,
        hooks=[gan_plotter_hook, tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
               tf.train.LoggingTensorHook([status_message], every_n_iter=100)],
        logdir=FLAGS.train_log_dir,
        get_hooks_fn=tfgan.get_joint_train_hooks())


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    tf.app.run()
