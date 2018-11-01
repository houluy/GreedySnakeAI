import tensorflow as tf
from collections import namedtuple

ConvLayers = namedtuple('ConvLayer',
    ('layer', 'type', 'shape', 'kernel', 'strides', 'number', 'channels', 'stddev', 'bias')
)
PoolLayers = namedtuple('PoolLayer', ('layer', 'type', 'shape', 'ksize', 'strides'))
# Local Response Normalizations
# LRNLayers = namedtuple('LRNLayer', ('layer', 'type', 'radius', 'bias', 'alpha', 'beta'))
FCLayers = namedtuple('FCLayer',
    ('layer', 'type', 'shape', 'stddev', 'bias', 'regularizer', 'regularizer_weight', 'activation')
)


class QApproximation:
    def __init__(self, ipt_size):
        self.ipt_size = ipt_size
        self.opt_size = 4
        self.channel = 1
        self.ipt = tf.placeholder(tf.float32, shape=(None, self.ipt_size, self.ipt_size, self.channel))
        self.hyper_params = {
            'alpha': 0.3
        }
        self.optimizer = tf.train.AdamOptimizer(self.alpha)

    @staticmethod
    def gen_weights(scope_name, shape, bias_shape, stddev=.1, bias=.1, regularizer=None, wl=None):
        weight_init = tf.truncated_normal_initializer(dtype=tf.float32, stddev=stddev)
        bias_init = tf.constant_initializer(bias)
        weights = tf.get_variable('{}-weights'.format(scope_name), shape=shape, initializer=weight_init)
        biases = tf.get_variable('{}-biases'.format(scope_name), shape=bias_shape, initializer=bias_init)
        if regularizer is not None:
            weights_loss = tf.multiply(tf.nn.l2_loss(weights), wl, name='weights-loss')
            tf.add_to_collection('losses', weights_loss)
        return weights, biases

    def _build_layer(self, ipt_layer, opt_layer):
        with tf.variable_scope(opt_layer.name, reuse=tf.AUTO_REUSE):
            if opt_layer.type != 'pool':
                weights, biases = self.gen_weights(opt_layer.name, opt_layer.shape, bias_shape=opt_layer.bias)
                if opt_layer.activation is not None:
                    return opt_layer.activation(tf.add(tf.matmul(ipt_layer, weights), biases))
            else:
                return tf.nn.max_pool(ipt_layer, ksize=opt_layer.ksize, strides=opt_layer.strides, padding='SAME')

    def build_all(self, structure):
        current = self.ipt
        for layer in structure:
            current = self._build_layer(current, layer)
        return current

    @property
    def loss(self): pass

    def __getattr__(self, name):
        if name in self.hyper_params:
            return self.hyper_params[name]
        else:
            raise AttributeError()

    @staticmethod
    def observe(game):
        return game.state

    @staticmethod
    def reward(game):
        if game.eat:
            return 10
        elif game.death:
            return -10
        else:
            return 0

    def train(self, sess): pass

