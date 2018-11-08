import tensorflow as tf
from collections import namedtuple
import numpy as np


class QApproximation:

    ConvLayers = namedtuple('ConvLayer',
        ('name', 'layer', 'kernel', 'strides', 'number', 'channels', 'stddev', 'bias')
    )
    PoolLayers = namedtuple('PoolLayer', ('name', 'layer', 'ksize', 'strides'))
    # Local Response Normalizations
    # LRNLayers = namedtuple('LRNLayer', ('layer', 'type', 'radius', 'bias', 'alpha', 'beta'))
    FCLayers = namedtuple('FCLayer',
        ('name', 'layer', 'shape', 'stddev', 'bias', 'regularizer', 'regularizer_weight', 'activation')
    )

    def __init__(self, ipt_size, out_size, ipt_channel=1):
        self.ipt_size = ipt_size
        self.ipt_shape = (self.ipt_size, self.ipt_size)
        self.ipt_channel = ipt_channel
        self.opt_size = out_size
        self.ipt = tf.placeholder(tf.float32, shape=(None, *self.ipt_shape, self.ipt_channel))
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
                self.ConvLayers(
                    name=name,
                    layer=1,
                    kernel=(3, 3),
                    strides=c_strides,
                    number=16,
                    channels=self.ipt_channel,
                    stddev=stddev,
                    bias=biases
                ),
                self.PoolLayers(
                    name=name,
                    layer=2,
                    ksize=k_size,
                    strides=p_strides,
                ),
                self.ConvLayers(
                    name=name,
                    layer=3,
                    kernel=(3, 3),
                    strides=c_strides,
                    number=32,
                    channels=16,
                    stddev=stddev,
                    bias=biases
                ),
                self.PoolLayers(
                    name=name,
                    layer=4,
                    ksize=k_size,
                    strides=p_strides,
                ),
                self.FCLayers(
                    name=name,
                    layer=5,
                    shape=128,
                    stddev=stddev,
                    bias=biases,
                    regularizer=True,
                    regularizer_weight=self.reg_lambda,
                    activation=tf.nn.relu,
                ),
                self.FCLayers(
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
        self._action = 0

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
            if isinstance(opt_layer, self.ConvLayers):
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
            elif isinstance(opt_layer, self.PoolLayers):
                clayer = tf.nn.max_pool(ipt_layer, ksize=opt_layer.ksize, strides=opt_layer.strides, padding='SAME')
            elif isinstance(opt_layer, self.FCLayers):
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

    @staticmethod
    def instant_reward(game):
        if game.eat:
            return 10
        elif game.death:
            return -10
        else:
            return 0

    def train(self, sess): pass


class DQN:

    exp = namedtuple('exp', ('state', 'action', 'reward', 'next_state'))

    def __init__(self, ipt_size, out_size):
        self.q_network = QApproximation(ipt_size, out_size)
        self.actions = list(range(out_size))
        self.experience_size = 1000
        self.experience_pool = []
        self.steps = 10000
        self.sess = tf.Session()
        self.hyper_params = {
            'alpha': 0.3,
            'epsilon': 0.3,
            'gamma': 0.9,
        }

    def gain_experiences(self, game):
        for _ in range(self.experience_size):
            state = self.observe(game)
            action_index = np.random.choice(self.actions)
            game.interact(action_index)
            reward = self.q.reward(game)
            next_state = self.observe(game)
            self.experience_pool.append(self.exp(
                state=state,
                action=action_index,
                reward=reward,
                next_state=next_state,
            ))
            if reward == -10:
                game.reset()

    def experience_replay(self):
        pass

    def train(self, game):
        game.reset()
        for step in range(self.steps):
            state = game.state
            epsilon = np.random.rand()
            action_index = self.epsilon_greedy(epsilon)
            game.interact(action_index)
            instant_reward = self.q.instant_reward(game)
            next_state, terminal = self.observe(game)
            if terminal:
                reward = instant_reward
            else:
                target = self.sess.run(self.target)
                reward = instant_reward + self.gamma *

            self.experience_pool.append(self.exp(
                state=state,
                action=action_index,
                reward=instant_reward,
                next_state=next_state,
            ))


    def epsilon_greedy(self, epsilon, next_state):
        if epsilon < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.sess.run(self.target, feed_dict={self.q_network.ipt: next_state})

    @property
    def action(self):
        return self._action

    @staticmethod
    def observe(game):
        return game.state, game.death

    @property
    def target(self):
        return self.q.networks['target']

    @property
    def q(self):
        return self.q.networks['approximation']

    def __getattr__(self, name):
        if name in self.hyper_params:
            return self.hyper_params[name]
        else:
            raise AttributeError()

    def __del__(self):
        self.sess.close()


if __name__ == '__main__':
    dqn = DQN(ipt_size=20, out_size=4)
