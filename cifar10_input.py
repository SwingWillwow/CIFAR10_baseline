import numpy as np
import tensorflow as tf
import os
from six.moves import xrange

# define global information
IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# return two tensor, one for image, another for label
def read_cifar10(filename_queue):
    # a class to store records
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    # set data information
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.channel = 3
    # count bytes for each image
    image_bytes = result.height * result.width * result.channel
    # bytes for each record
    record_bytes = image_bytes + label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    # get label
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # get image
    channel_major = tf.reshape(tf.strided_slice(record_bytes,
                                                [label_bytes],
                                                [label_bytes+image_bytes]),
                               [result.channel, result.height, result.width])
    # change [channel, height, width] to [height, width, channel]
    result.uint8image = tf.transpose(channel_major, [1, 2, 0])
    return result


# this method generate batch of images and labels
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


# distorted_inputs,those are distorted_images used to data augmentation
def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    # check if all the files have been establish
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # cyclic get filename
    filename_queue = tf.train.string_input_producer(filenames)

    with tf.name_scope('data_augmentation'):
        # read image and label from .bin files
        read_input = read_cifar10(filename_queue) # read data from i.bin
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        # decide image size, we just use 24x24 area of the 32x32 cifar-data
        height = IMAGE_SIZE
        width = IMAGE_SIZE
        # crop picture
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
        # flip the image horizontally
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # randomly change the brightness of the picture
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        # randomly change the contrast of the image
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        # standardization the image
        float_image = tf.image.per_image_standardization(distorted_image)

        # set the shape of tensor
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d CIFAR images before starting to train.'
              'This will take a few minutes.' % min_queue_examples)

        # generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


# this method generate inputs for both training and evaluating
def inputs(eval_data, data_dir, batch_size):
    # use eval_data to define whether the data used to train or evaluate
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # check file exist
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames)

        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               height, width)
        float_iamge = tf.image.per_image_standardization(resized_image)

        float_iamge.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_iamge, read_input.label, min_queue_examples,
                                           batch_size, True)




