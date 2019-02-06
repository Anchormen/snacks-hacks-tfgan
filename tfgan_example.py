import tensorflow as tf
tfgan = tf.contrib.gan

# Set up the input pipeline.
images = mnist_data_provider.provide_data(FLAGS.batch_size)
noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

# Manually build the generator and discriminator.
with tf.variable_scope('Generator') as gen_scope:
    generated_images = generator_fn(noise)
with tf.variable_scope('Discriminator') as dis_scope:
    discriminator_gen_outputs = discriminator_fn(generated_images)
with variable_scope.variable_scope(dis_scope, reuse=True):
    discriminator_real_outputs = discriminator_fn(images)

generator_variables = variables_lib.get_trainable_variables(gen_scope)
discriminator_variables = variables_lib.get_trainable_variables(dis_scope)

# Depending on what TF-GAN features you use, you don't always need to supply
# every `GANModel` field. At a minimum, you need to include the discriminator
# outputs and variables if you want to use TF-GAN to construct losses.
gan_model = tfgan.GANModel(
    generator_inputs,
    generated_data,
    generator_variables,
    gen_scope,
    generator_fn,
    real_data,
    discriminator_real_outputs,
    discriminator_gen_outputs,
    discriminator_variables,
    dis_scope,
    discriminator_fn)

# Build the GAN loss.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)

# Create the train ops, which calculate gradients and apply updates to weights.
train_ops = tfgan.gan_train_ops(
    gan_model,
    gan_loss,
    generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5))

# Run the train ops in the alternating training scheme.
tfgan.gan_train(
    train_ops,
    hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps)],
    logdir=FLAGS.train_log_dir)