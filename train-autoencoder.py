import time
import argparse
import os
import tensorflow as tf
import numpy as np
import _pickle as pkl
from scipy.misc import imsave as ims
from scipy.misc import toimage

from utils import *

from Autoencoder import Autoencoder
from tensorflow.examples.tutorials.mnist import input_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='mnist',
                       help='data file name')
    parser.add_argument('--file_path', type=str, default='../data/',
                       help='path to files')
    parser.add_argument('--save_dir', type=str, default='mvs/ae',
                       help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type = str, default = 'mvs/ae_log',
                      help = 'directory for logging')
    parser.add_argument('--img_size_w', type=int, default=28,
                       help='size of input images')
    parser.add_argument('--img_size_h', type=int, default=28,
                      help='size of input images')
    parser.add_argument('--n_channels', type=int, default=1,
                      help='number of channels of input images')
    parser.add_argument('--n_samples', type=None, default=None,
                       help='number of samples')
    parser.add_argument('--n_input', type=int, default=28*28*1,
                      help='dim of input')
    parser.add_argument('--n_z', type=int, default=2,
                      help='dim of latents')
    parser.add_argument('--h1', type=int, default=500,
                      help='dim of hidden layer 1')
    parser.add_argument('--h2', type=int, default=500,
                      help='dim of  hidden layer 2')
    parser.add_argument('--model', type=str, default='conv',
                       help='architecture -- can be linear, nonlinear or conv')
    parser.add_argument('--vae',action='store_true', help='vae or normal autoencoder')
    parser.add_argument('--loss', type=str, default='squared_error',
                       help='squared_error or bernoulli')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='minibatch size')
    parser.add_argument('--num_epochs_ae', type=int, default=150,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=10000,
                       help='save frequency')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='learning rate')
    parser.add_argument('--weight_recon_loss', type=float, default=1,
                       help='weight reconstruction loss')
    parser.add_argument('--weight_lat_loss', type=float, default=1,
                      help='weight latent loss')
    parser.add_argument('--rla_annealing',action='store_true',
                            help = 'Do reconstruction loss annealing')
    parser.add_argument('--rla_rate_rise_factor', type = float, default = 0.001,
                        help='recon loss weight is increasd by this much every save_every steps')
    parser.add_argument('--rla_rate_rise_time', type = int, default = 10,
                        help = 'iterations before increasing recon loss term')
    parser.add_argument('--decay_rate', type=float, default=1,
                       help='decay rate for learning rate')
    parser.add_argument('--img_data', type = bool, default = True,
                            help = 'true if image data')
    parser.add_argument('--normalize',action='store_true', help = 'normalize')
    args = parser.parse_args()
    modelparam = make_modelparam_string(args)
    train(args, modelparam)


def train(args, modelparam = ""):

    # Load MNIST-M
    # mnist = pkl.load(open(os.path.join(args.file_path,args.data_set), 'rb'))

    mnist = input_data.read_data_sets('MNIST_data')
    mnist_train_images = mnist.train.images
    mnist_test_images = mnist.test.images

    # save configuration
    save_dir_modelparam = os.path.join(args.save_dir, modelparam)
    if not os.path.isdir(save_dir_modelparam):
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        os.makedirs(save_dir_modelparam)
    with open(os.path.join(save_dir_modelparam, 'config.pkl'), 'wb') as f:
        pkl.dump(args, f)

    model = Autoencoder(args)
    summariesAutoEncOp = tf.summary.merge(model.summaries)

    with tf.Session() as sess:
            # init
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            print("Trainable")
            names = [x.name for x in tf.trainable_variables()]
            [print(n) for n in names]

            start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

            # summary writer
            writer = tf.summary.FileWriter(os.path.join(args.log_dir,
                                                        modelparam,
                                                        start_time),
                                            sess.graph)

            tf.get_default_graph().finalize()

            # Batch generators
            gen_train_batch = batch_generator([mnist_train_images], args.batch_size)

            visualization = next(gen_train_batch)
            visualization = visualization[0]
            visualization = np.squeeze(visualization.reshape([args.batch_size, args.img_size_h, args.img_size_w, args.n_channels]))
            num_imgs = int(np.floor(np.sqrt(visualization.shape[0])))
            im = toimage(merge(visualization[:(num_imgs*num_imgs)],[num_imgs,num_imgs]))
            im.save(os.path.join(save_dir_modelparam,"base_train"+modelparam+".jpg"))

            # Batch generators
            gen_test_batch = batch_generator([mnist_test_images], args.batch_size)

            visualization = next(gen_test_batch)
            visualization = visualization[0]
            visualization = np.squeeze(visualization.reshape([args.batch_size, args.img_size_h, args.img_size_w, args.n_channels]))
            im = toimage(merge(visualization[:(num_imgs*num_imgs)],[num_imgs,num_imgs]))
            im.save(os.path.join(save_dir_modelparam,"base_test"+modelparam+".jpg"))

            num_batches = int(len(mnist_train_images)/args.batch_size)

                # run training
            for e in range(args.num_epochs_ae):
                print("\nEpoch " + str(e))

                new_lr = model.lr.eval() * (args.decay_rate ** e)
                sess.run(model.lr_update, feed_dict={model.new_lr: new_lr})
                print("Learning rate: " + str(model.lr.eval()))

                for b in range(num_batches):
                    global_step = e * num_batches + b

                    X = next(gen_train_batch)
                    batch = X[0].reshape(args.batch_size, args.img_size_h*args.img_size_w*args.n_channels)

                    start = time.time()

                    _, gen_loss, lat_loss, s = sess.run([model.train_op,
                                                     model.recon_loss,
                                                     model.lat_loss,
                                                     summariesAutoEncOp],
                                                     feed_dict={model.images_in: batch})

                    end = time.time()
                    writer.add_summary(s, global_step)
                    print("{}/{} (epoch {}), gen_loss = {:.3f}, lat_loss = {:.3f},\
                                time/batch = {:.3f}"
                    .format(global_step,
                                args.num_epochs_ae * num_batches,
                                e, np.mean(gen_loss), np.mean(lat_loss), end - start))

                    # save model and visualize
                    if global_step % args.save_every == 0 or (e==args.num_epochs_ae-1 and b == num_batches-1): # save for the last result
                        checkpoint_path = os.path.join(args.log_dir,
                                                       modelparam,
                                                       start_time,
                                                       'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step = global_step)
                        print("model saved to {}".format(checkpoint_path))
                        # viz

                        generated_test= sess.run(model.generated_images_out,
                                                  feed_dict={model.images_in: visualization.reshape(args.batch_size, args.n_input)})
                        generated_test = np.squeeze(generated_test)
                        im = toimage(merge(generated_test[:(num_imgs*num_imgs)],[num_imgs,num_imgs]))
                        im.save(os.path.join(save_dir_modelparam,str(e)+modelparam+".jpg"))





if __name__ == '__main__':
    main()
