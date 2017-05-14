import tensorflow as tf
import argparse
import os
from six.moves import cPickle
from scipy.misc import imsave as ims
from scipy.misc import toimage

from Autoencoder import Autoencoder
from utils import *
from viz_utils import *
import matplotlib.pyplot as plt
import re
import _pickle as pkl
from tensorflow.examples.tutorials.mnist import input_data

s_dir = 'mvs2/ae'
l_dir = 'mvs2/ae_log'
strin = "vae_False,mod_conv,nz_2,lr_1E-03,wr_1E+00,wp_1E+00,dr_1E+00,num_ep150,d_mnist"
dirs = get_dirs2(s_dir, l_dir, strin)
save_dir = dirs[0]
ckpt_dir = dirs[1]
n_show_reconstruction = 5
print(save_dir)
print(ckpt_dir)

data_sets = ['mnist']

with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
    saved_args = cPickle.load(f)


save_dir_train = os.path.join(save_dir, 'train')
if not os.path.isdir(save_dir_train):
    os.makedirs(save_dir_train)
save_dir_test = os.path.join(save_dir, 'test')
if not os.path.isdir(save_dir_test):
    os.makedirs(os.path.join(save_dir_test))

model = Autoencoder(saved_args)
# sess = tf.InteractiveSession()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, os.path.join(ckpt_dir, ckpt.model_checkpoint_path.split("/")[-1]))

        for data_set in data_sets:
            # Load data set
            # mnistm = pkl.load(open(os.path.join(saved_args.file_path, data_set), 'rb'))

            mnist = input_data.read_data_sets('MNIST_data')
            mnist_train_images = mnist.train.images
            mnist_test_images = mnist.test.images

            train_samp = mnist_train_images[:saved_args.batch_size]
            test_samp = mnist_test_images[:saved_args.batch_size]

            # training set
            train_samp_img = np.squeeze(train_samp.reshape(saved_args.batch_size,
                        saved_args.img_size_h,saved_args.img_size_w,saved_args.n_channels))
            # reconstruct
            reconstructed_images_train = model.reconstruct(sess, train_samp)
            reconstructed_images_train = np.squeeze(reconstructed_images_train.reshape(saved_args.batch_size,
                                                                saved_args.img_size_h,
                                                                saved_args.img_size_w,
                                                                saved_args.n_channels))
            num_imgs = int(np.floor(np.sqrt(reconstructed_images_train.shape[0])))
            im = toimage(merge(reconstructed_images_train[:(num_imgs*num_imgs)],[num_imgs,num_imgs]))
            im.save(os.path.join(save_dir_train,data_set+"reconstructed_images_train.jpg"))
            im = toimage(merge(train_samp_img[:(num_imgs*num_imgs)],[num_imgs,num_imgs]))
            im.save(os.path.join(save_dir_train,data_set+"original_images_train.jpg"))

            plt.figure(figsize=(8, 12))
            for i in range(n_show_reconstruction):
                plt.subplot(n_show_reconstruction, 2, 2*i + 1)
                plt.imshow(train_samp_img[i], vmin=0, vmax=1, cmap="gray")
                plt.title("Training data input")
                plt.colorbar()
                plt.subplot(n_show_reconstruction, 2, 2*i + 2)
                plt.imshow(reconstructed_images_train[i], vmin=0, vmax=1, cmap="gray")
                plt.title("Reconstruction")
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir_train,data_set+"reconstructed_images_train_sbs.jpg"))


            # test set
            test_samp_img = np.squeeze(test_samp.reshape(saved_args.batch_size,
                        saved_args.img_size_h,saved_args.img_size_w,saved_args.n_channels))
            # reconstruct
            reconstructed_images_flat = model.reconstruct(sess, test_samp)
            reconstructed_images_flat = np.squeeze(reconstructed_images_flat.reshape(saved_args.batch_size,
                                                                saved_args.img_size_h,
                                                                saved_args.img_size_w,
                                                                saved_args.n_channels))
            im = toimage(merge(reconstructed_images_flat[:(num_imgs*num_imgs)],[num_imgs,num_imgs]))
            im.save(os.path.join(save_dir_test,data_set+"reconstructed_images_test.jpg"))
            im = toimage(merge(test_samp_img[:(num_imgs*num_imgs)],[num_imgs,num_imgs]))
            im.save(os.path.join(save_dir_test,data_set+"original_images_test.jpg"))


            plt.figure(figsize=(8, 12))
            for i in range(n_show_reconstruction):
                plt.subplot(n_show_reconstruction, 2, 2*i + 1)
                plt.imshow(test_samp_img[i], vmin=0, vmax=1, cmap="gray")
                plt.title("Test data input")
                plt.colorbar()
                plt.subplot(n_show_reconstruction, 2, 2*i + 2)
                plt.imshow(reconstructed_images_flat[i], vmin=0, vmax=1, cmap="gray")
                plt.title("Reconstruction")
                plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir_test,data_set+"reconstructed_images_test_sbs.jpg"))


            nx = ny = 30
            canvas = np.empty((saved_args.img_size_h*ny, saved_args.img_size_w*nx))

            if saved_args.n_z == 2:

                x_values = np.linspace(-2, 2, nx)
                y_values = np.linspace(-2, 2, ny)

                for i, yi in enumerate(x_values):
                    for j, xi in enumerate(y_values):
                        z_mu = np.array([[xi, yi]]*saved_args.batch_size)
                        x_mean = model.generate(sess, saved_args, saved_args.batch_size, z_mu)
                        canvas[(nx-i-1)*saved_args.img_size_w:(nx-i)*saved_args.img_size_w,
                               j*saved_args.img_size_h:(j+1)*saved_args.img_size_h] = x_mean[0].reshape(saved_args.img_size_h, saved_args.img_size_w)

                plt.figure(figsize=(8, 10))
                Xi, Yi = np.meshgrid(x_values, y_values)
                plt.imshow(canvas, origin="upper", cmap="gray")
                plt.tight_layout()
                ims(os.path.join(save_dir_test,"generated_images_sampled_latent_space.jpg"),canvas)




    else:
        print("No checkpoint file found.")
