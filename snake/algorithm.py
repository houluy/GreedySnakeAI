import tensorflow as tf
from collections import namedtuple

ConvLayers = namedtuple('ConvLayer',
    ('name', 'layer', 'kernel', 'strides', 'number', 'channels', 'stddev', 'bias')
)
PoolLayers = namedtuple('PoolLayer', ('name', 'layer', 'ksize', 'strides'))
# Local Response Normalizations
# LRNLayers = namedtuple('LRNLayer', ('layer', 'type', 'radius', 'bias', 'alpha', 'beta'))
FCLayers = namedtuple('FCLayer',
    ('name', 'layer', 'shape', 'stddev', 'bias', 'regularizer', 'regularizer_weight', 'activation')
)


class QApproximation:
    def __init__(self, ipt_size, ipt_channel=1):
        self.ipt_size = ipt_size
        self.ipt_shape = (self.ipt_size, self.ipt_size)
        self.ipt_channel = ipt_channel
        self.opt_size = 4
        self.ipt = tf.placeholder(tf.float32, shape=(None, *self.ipt_shape, self.ipt_channel))
        self.hyper_params = {
            'alpha': 0.3
        }
        self.reg_lambda = 0.03
        self.optimizer = tf.train.AdamOptimizer(self.alpha)

        c_strides = (1, 1, 1, 1)
        p_strides = (1, 2, 2, 1)
        k_size = (1, 3, 3, 1)
        stddev = 5e-2
        biases = 0.1
        layer_name = ['approximation', 'target']
        self.networks = {}
        for name in layer_name:
            self.networks[name] = self.build_all([
                ConvLayers(
                    name=name,
                    layer=1,
                    kernel=(3, 3),
                    strides=c_strides,
                    number=16,
                    channels=self.ipt_channel,
                    stddev=stddev,
                    bias=biases
                ),
                PoolLayers(
                    name=name,
                    layer=2,
                    ksize=k_size,
                    strides=p_strides,
                ),
                ConvLayers(
                    name=name,
                    layer=3,
                    kernel=(3, 3),
                    strides=c_strides,
                    number=32,
                    channels=16,
                    stddev=stddev,
                    bias=biases
                ),
                PoolLayers(
                    name=name,
                    layer=4,
                    ksize=k_size,
                    strides=p_strides,
                ),
                FCLayers(
                    name=name,
                    layer=5,
                    shape=128,
                    stddev=stddev,
                    bias=biases,
                    regularizer=True,
                    regularizer_weight=self.reg_lambda,
                    activation=tf.nn.relu,
                ),
                FCLayers(
                    name=name,
                    layer=6,
                    shape=4,
                    stddev=stddev,
                    bias=biases,
                    regularizer=None,
                    regularizer_weight=self.reg_lambda,
                    activation=None,
                )
            ])

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
            if isinstance(opt_layer, ConvLayers):
                weight_shape = [*opt_layer.kernel, opt_layer.channels, opt_layer.number]
                weights, biases = self.gen_weights(
                    opt_layer.name + str(opt_layer.layer),
                    weight_shape,
                    bias_shape=[opt_layer.number],
                    stddev=opt_layer.stddev,
                    bias=opt_layer.bias,
                )
                clayer = tf.nn.conv2d(ipt_layer, weights, strides=opt_layer.strides, padding='SAME')
                clayer = tf.nn.relu(tf.nn.bias_add(clayer, biases))
            elif isinstance(opt_layer, PoolLayers):
                clayer = tf.nn.max_pool(ipt_layer, ksize=opt_layer.ksize, strides=opt_layer.strides, padding='SAME')
            elif isinstance(opt_layer, FCLayers):
                ipt_layer = tf.layers.Flatten()(ipt_layer)
                ipt_size = ipt_layer.get_shape()[-1]
                weight_shape = [ipt_size, opt_layer.shape]
                weights, biases = self.gen_weights(
                    opt_layer.name + str(opt_layer.layer),
                    weight_shape,
                    bias_shape=[opt_layer.shape],
                    regularizer=opt_layer.regularizer,
                    wl=opt_layer.regularizer_weight,
                )
                clayer = tf.add(tf.matmul(ipt_layer, weights), biases)
                if opt_layer.activation is not None:
                    clayer = opt_layer.activation(clayer)
            return clayer

    def build_all(self, structure):
        current = self.ipt
        for layer in structure:
            current = self._build_layer(current, layer)
        return tf.nn.softmax(current)

    @property
    def loss(self): pass

    @property
    def action(self): pass

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


if __name__ == '__main__':
    qa = QApproximation(20)
    print(qa.networks)
