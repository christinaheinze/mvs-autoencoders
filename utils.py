import numpy as np
import tensorflow as tf

def merge(images, size):
    if images.ndim == 4:
        h, w, d = images.shape[1], images.shape[2], images.shape[3]
        img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if images.ndim == 4:
            img[j*h:j*h+h, i*w:i*w+w,:] = image
        else:
            img[j*h:j*h+h, i*w:i*w+w] = image
    return img

def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]

def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def make_modelparam_string(args):
    return "vae_%s,mod_%s,nz_%d,lr_%.0E,wr_%.0E,wp_%.0E,dr_%.0E,num_ep%d,d_%s" % (args.vae,
                                     args.model,
                                     args.n_z,
                                     args.learning_rate,
                                     args.weight_recon_loss,
                                     args.weight_lat_loss,
                                     args.decay_rate,
                                     args.num_epochs_ae,
                                     args.data_set)
