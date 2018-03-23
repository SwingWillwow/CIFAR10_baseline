# build-in
import os
import re
import sys
import tarfile
from six.moves import urllib
# third-party
import tensorflow as tf
import cifar10_input


FLAGS = tf.app.flags.FLAGS
# set sth global variables
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch""")
tf.app.flags.DEFINE_string('data_dir', 'tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Training the model use fp16""")

# constant describe the cifar-10 data set
# basically just extract information defined in cifar10_input

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLE_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLE_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# constant describing the training process

MOVING_AVERAGE_DECAY = 0.9999     # the decay to use for moving average
NUM_EPOCH_PER_DECAY = 350.0       # EPOCH after which learning rate decay
LEARNING_RATE_DECAY_FACTOR = 0.1  # learning rate decay factor
INITIAL_LEARNING_RATE = 0.1       # initial learning rate


# prefix for multiple GPUs training
TOWER_NAME = 'tower'
# url to download cifar-10
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    """help to create summary of activation, including histogram and sparsity"""
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    # histogram for activations
    tf.summary.histogram(tensor_name + '/activations', x)
    # sparsity of activations, simply do zero counts div tensor size
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Help to create a variable stored on CPU Memory"""
    # first cpu
    with tf.device('/cpu:0'):
        # set dtype
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    create an variable with l2 norm penalty
    basically use to create a weight parameter in nn
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        # add l2 norm to losses
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    """ Construct distorted input for CIFAR training using the Reader ops.
        Those are data generate by distort original picture in order to augment data.
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a directory.')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)
    # if use float16 just cast
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    # return data
    return images, labels


def inputs(eval_data):
    """
    This function can generate data for both training and evaluating.

    :param eval_data: Boolean, True to generate evaluating data. else training.

    :return:
    """
    # check is there has a directory
    if not FLAGS.data_dir:
        raise ValueError('Please supply a directory.')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data,
                                          data_dir,
                                          FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images):
    """the cifar-10 baseline model.


    Args:
        images: Images returned from distorted_input() or input().

    Return:
        logits. we don't compute softmax because
        tf.nn.sparse_softmax_cross_entropy_with_logits() function compute soft max
        internally for efficiency.
    """
    # defined conv1
    with tf.variable_scope('conv1') as scope:
        # aka filter
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        bias = _variable_on_cpu('biases',
                                [64],
                                tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, bias)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # summary conv1's activations information
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # norm1 local response normalization, this help to improve performance

    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2 just same as conv1
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        bias = _variable_on_cpu('biases', [64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, bias)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    # pool2

    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # local3 (fully-connected layer) also known as dense layer

    with tf.variable_scope('local3') as scope:
        # reshape to flatten shape:[batch_size, flatten_len].
        # thus, we can deal with the fc-layer in a matrix multiple way.
        flatten = tf.reshape(pool2, [images.get_shape()[0], -1])
        # get dimension per input. flatten's shape [batch, input_size]
        dim = flatten.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        bias = _variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
        local3 = tf.nn.relu(tf.matmul(flatten, weights) + bias, name=scope.name)
        _activation_summary(local3)

    # local4

    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        bias = _variable_on_cpu('biases', [192], initializer=tf.constant_initializer(0.0))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + bias, name=scope.name)
        _activation_summary(local4)

    # simple linear layer y = Wx + b without non-linearity Relu
    with tf.variable_scope('logit') as scope:
        weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES],
                                              stddev=0.04, wd=0.004)
        bias = _variable_on_cpu('biases', [NUM_CLASSES],
                                initializer=tf.constant_initializer(0.0))
        logit = tf.add(tf.matmul(local4, weights), bias, name=scope.name)
        _activation_summary(logit)
    return logit


def loss(logits, labels):
    """Add L2Loss to all trainable variables except biases"""
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits,
                                                                   name='cross_entropy_per_example')
    # mean the cross entropy
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # this add_n operation help us get total loss
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """
    Generate Moving average for all losses and associated summaries for
    visualizing the performance of the network.

    """

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    # this op count ExponentialMovingAverage for all variables in list
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    # num batches / epoch
    num_batches_per_epoch = NUM_EXAMPLE_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    # count how many step to decay the learning rate
    decay_steps = int(num_batches_per_epoch * NUM_EPOCH_PER_DECAY)
    # get a exponential decay learning rate
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    # generate moving averages of all losses and associated summaries
    loss_averages_op = _add_loss_summaries(total_loss)

    # compute gradients, control_dependencies help us to ensure last loss information
    # has been processed
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        # just a placeholder can define this op later
        train_op = tf.no_op(name='train')
    return train_op


def maybe_download_and_extract():
    """
    use this function to get cifar-10 data set from URL or file system

    :return:
    """
    dest_directory = FLAGS.data_dir
    # check if the path exist
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # use this function to see how the progress of downloading
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        # download files
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        # print success information
        print('Successfully downloaded ', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    # extract files from .gz package
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
