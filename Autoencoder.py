import tensorflow as tf
import numpy as np
from ops import *
from utils import *

class Autoencoder():
    def __init__(self, args):
        self.args = args
        self.summaries = list()
        with tf.name_scope("input"):

            self.images_in = tf.placeholder(tf.float32, \
                                [self.args.batch_size, self.args.n_input], name = "images")
            if self.args.normalize:
                self.images = (tf.cast(self.images_in, tf.float32)) / 255
            else:
                self.images = self.images_in

            if self.args.img_data:
                self.image_matrix = tf.reshape(self.images,[-1, self.args.img_size_h,
                                            self.args.img_size_w,
                                            self.args.n_channels])
                self.summaries.append(tf.summary.image('input', self.image_matrix, 6))


        with tf.name_scope("recognition"):
            if self.args.model == "nonlinear":
                guessed_z_tmp = self.recognition_nonlin(self.images, vae = self.args.vae)
            elif self.args.model == "conv":
                guessed_z_tmp = self.recognition_conv(self.image_matrix, vae = self.args.vae)
            else:
                guessed_z_tmp = self.recognition(self.images, vae = self.args.vae)

            if self.args.vae:
                self.z_mean = guessed_z_tmp[0]
                self.z_log_sigma_sq = guessed_z_tmp[1]
            else:
                self.guessed_z = guessed_z_tmp

        with tf.name_scope("generation"):
            if self.args.vae:
                samples = tf.random_normal([self.args.batch_size,self.args.n_z],0,1,dtype=tf.float32)
                self.guessed_z = tf.add(self.z_mean,
                                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)),
                                        samples))

            if self.args.model == "nonlinear":
                self.generated_images = self.generation_nonlin(self.guessed_z)
            elif self.args.model == "conv":
                self.generated_images = self.generation_conv(self.guessed_z)
            else:
                self.generated_images = self.generation(self.guessed_z)

            if self.args.normalize:
                self.generated_images_out = self.generated_images*255
            else:
                self.generated_images_out = self.generated_images

            if self.args.img_data:
                self.summaries.append(tf.summary.image('generated_samples_out',
                                                        self.generated_images_out, 6))

            self.generated_flat = tf.reshape(self.generated_images, [-1, self.args.n_input])

        with tf.name_scope("loss"):
            if self.args.rla_annealing:
                self.rla_rate = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            else:
                self.rla_rate = tf.Variable(1.0, trainable=False, dtype=tf.float32)
            self.new_rla_rate = tf.placeholder(tf.float32, shape=[], name="new_rla_rate")
            self.rla_rate_update = tf.assign(self.rla_rate, self.new_rla_rate)

            if self.args.loss == "squared_error":
                self.recon_loss = self.rla_rate * \
                                    0.5 * tf.reduce_sum(tf.pow(tf.subtract(\
                                    self.generated_flat,self.images), 2.0),1)
            else:
                self.recon_loss = \
                     -tf.reduce_sum(self.images * \
                     tf.log(1e-10 + self.generated_flat)+\
                      (1-self.images) * \
                      tf.log(1e-10 + 1 - self.generated_flat),1)

            if self.args.vae:
                self.lat_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) +
                                                     tf.exp(self.z_log_sigma_sq) -
                                                     self.z_log_sigma_sq - 1,1)
            else:
                self.lat_loss = tf.zeros(self.args.batch_size)

            self.cost = tf.reduce_mean(self.args.weight_recon_loss * self.recon_loss+ \
                                        self.args.weight_lat_loss * self.lat_loss)

            self.summaries.append(tf.summary.scalar("recon_loss", tf.reduce_mean(self.recon_loss)))
            self.summaries.append(tf.summary.scalar("lat_loss",  tf.reduce_mean(self.lat_loss)))
            self.summaries.append(tf.summary.scalar("cost", self.cost))

        with tf.name_scope("optimizer"):
            self.lr = tf.Variable(self.args.learning_rate, trainable=False, dtype=tf.float32)
            self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
            self.lr_update = tf.assign(self.lr, self.new_lr)
            opt_func = tf.train.AdamOptimizer(self.lr)
            # tvars = tf.trainable_variables()
            #
            # self.grads = tf.gradients(self.cost, tvars)
            # self.grad_clipped = tf.clip_by_global_norm(self.grads,1)
            # self.train_op = opt_func.apply_gradients(zip(self.grad_clipped, tvars))
            self.train_op = opt_func.minimize(self.cost)

    def recognition(self, input_images, vae = False, reuse = None):

        if vae:
            w_mean = tf.layers.dense(input_images,
                                     units = self.args.n_z,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name = "w_mean",
                                     reuse=reuse)
            w_stddev = tf.layers.dense(input_images,
                                       units = self.args.n_z,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name = "w_stddev",
                                       reuse=reuse)

            tf.summary.histogram("weights", w_mean)
            tf.summary.histogram("weights_sd", w_stddev)

            return w_mean, w_stddev

        else:
            self.z = tf.layers.dense(input_images,
                                     units = self.args.n_z,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name = "z",
                                     reuse=reuse)

            return self.z

    def recognition_nonlin(self, input_images, vae = False, reuse = None):
        h1 = tf.layers.dense(input_images,
                                 units = self.args.h1,
                                 activation=tf.nn.softplus,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name = "h1_rec",
                                 reuse=reuse)
        h2 = tf.layers.dense(h1,
                             units = self.args.h2,
                             activation=tf.nn.softplus,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.zeros_initializer(),
                             name = "h2_rec",
                             reuse=reuse)


        if vae:
            w_stddev = tf.layers.dense(h2, units = self.args.n_z, name = "w_stddev")
            w_mean = tf.layers.dense(h2,
                                units = self.args.n_z,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(),
                                name = "w_mean",
                                reuse=reuse)
            return  w_mean, w_stddev
        else:
            z = tf.layers.dense(h2,
                                units = self.args.n_z,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(),
                                name = "z",
                                reuse=reuse)
            return z

    def recognition_conv(self, input_images, vae = False, reuse = None):
        h1 = tf.layers.conv2d(input_images,
                                  filters = 16,
                                  kernel_size=[5, 5],
                                  strides=2,
                                  padding="same",
                                  activation=lrelu,
                                  name = "rec_conv_h1",
                                  reuse=reuse) # 28x28x3 -> 14x14x16

        h2 = tf.layers.conv2d(h1,
                                  filters = 32,
                                  kernel_size=[5, 5],
                                  strides=2,
                                  padding="same",
                                  activation=lrelu,
                                  name = "rec_conv_h2",
                                  reuse=reuse) # 14x14x16 -> 7x7x32
        h2_flat = tf.reshape(h2,[-1, 7*7*32])

        if vae:
            w_mean = tf.layers.dense(h2_flat, units = self.args.n_z, name = "w_mean",reuse=reuse)
            w_stddev = tf.layers.dense(h2_flat, units = self.args.n_z, name = "w_stddev",reuse=reuse)
            return  w_mean, w_stddev
        else:
            z = tf.layers.dense(h2_flat, units = self.args.n_z, name = "z_rec",reuse=reuse)
            return z

    # decoder
    def generation_conv(self, z, reuse = None):
        z_develop = tf.layers.dense(z, units = 7*7*32, reuse=reuse)
        z_matrix = tf.nn.relu(tf.reshape(z_develop, [-1, 7, 7, 32]))
        h1 = tf.layers.conv2d_transpose(z_matrix,
                                        kernel_size=[5, 5],
                                        filters = 16,
                                        strides=2,
                                        padding="same",
                                        activation=tf.nn.relu,
                                        name = "g_h1",
                                        reuse=reuse)
        h2 = tf.layers.conv2d_transpose(h1,
                                        kernel_size=[5, 5],
                                        filters = self.args.n_channels,
                                        strides=2,
                                        padding="same",
                                        name = "g_h2",
                                        reuse=reuse)
        h2 = tf.nn.sigmoid(h2)

        return h2


    # decoder
    def generation(self, z, reuse = None):
        z_develop = tf.layers.dense(z,
                        units = self.args.n_input,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        name = "z_gen",
                        reuse=reuse)

        if self.args.img_data:
            z_develop = tf.nn.sigmoid(z_develop)
            z_develop = tf.reshape(z_develop, [-1,  self.args.img_size_h,
                                                    self.args.img_size_w,
                                                    self.args.n_channels])

        return z_develop

    def generation_nonlin(self, z, reuse = None):
        h1 = tf.layers.dense(z,
                             units = self.args.h1,
                             activation=tf.nn.softplus,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name = "h1_gen",
                             reuse=reuse)
        h2 = tf.layers.dense(h1,
                             units = self.args.h2,
                             activation=tf.nn.softplus,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name = "h2_gen",
                             reuse=reuse)

        z_develop = tf.layers.dense(h1,
                                    units = self.args.n_input,
                                    name = "z_gen",
                                    reuse=reuse)
        if self.args.img_data:
            z_develop = tf.nn.sigmoid(z_develop)
            z_develop = tf.reshape(z_develop, [-1, self.args.img_size_h,
                                                   self.args.img_size_w,
                                                   self.args.n_channels])
        return z_develop


    def generate(self, sess, saved_args, n, z_mu=None):
        """ Generate data by sampling from latent space.  """
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        if z_mu is None:
            z_mu = np.random.normal(size=(n, saved_args.n_z))
        return sess.run(self.generated_images,
                             feed_dict={self.guessed_z: z_mu})

    def reconstruct(self, sess, images):
        """ Use VAE to reconstruct given data. """
        return sess.run(self.generated_images_out, feed_dict={self.images_in: images})
