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

    def __init__(self, ipt_size, out_size, batch_size, ipt_channel=1):
        self.ipt_size = ipt_size
        self.ipt_shape = (self.ipt_size, self.ipt_size)
        self.ipt_channel = ipt_channel
        self.batch_size = batch_size
        self.opt_size = out_size
        self.ipt = tf.placeholder(tf.float32, shape=(self.batch_size, *self.ipt_shape, self.ipt_channel))
        self.reward = tf.placeholder(tf.float32, shape=(self.batch_size, 1))

        # Below is the output mask that only the chosen neural (action) will output Q.
        self.opt_mask = tf.placeholder(tf.float32, shape=(self.batch_size, self.opt_size, 1))
        self.reg_lambda = 0.03
        self.alpha = 0.03
        self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

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
                    shape=self.opt_size,
                    stddev=stddev,
                    bias=biases,
                    regularizer=None,
                    regularizer_weight=self.reg_lambda,
                    activation=None,
                )
            ], name=name)
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

    def build_all(self, structure, name):
        current = self.ipt
        for layer in structure:
            current = self._build_layer(current, layer)
        if name == 'approximation':
            return tf.matmul(current, self.opt_mask)
        else:
            return current

    @property
    def loss(self):
        return tf.reduce_mean(tf.square(self.reward - self.networks['approximation']))

    @property
    def action(self): pass

    def train(self, sess): pass


class DQN:

    exp = namedtuple('exp', ('state', 'action', 'reward', 'next_state', 'terminal'))

    def __init__(self, ipt_size, out_size):
        self.q_network = QApproximation(ipt_size, out_size)
        self.actions = list(range(out_size))
        self.experience_size = 0
        self.experience_pool = []
        self.steps = 1000
        self.episodes = 10000
        self.minibatch_size = 128
        self.target_update_episode = 10
        self.sess = tf.Session()
        self.hyper_params = {
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

    def _convert(self, minibatch):
        'Convert minibatch from namedtuple to multi-dimensional matrix'
        for block in minibatch:
            pass
        return minibatch

    def train(self, game):
        game.reset()
        for episode in range(self.episodes):
            state = game.state
            for step in range(self.steps):
                epsilon = np.random.rand()
                action_index = self.epsilon_greedy(epsilon)
                game.interact(action_index)
                next_state, instant_reward, terminal = self.observe(game)
                self.experience_pool.append(self.exp(
                    state=state,
                    action=action_index,
                    reward=instant_reward,
                    next_state=next_state,
                    terminal=terminal,
                ))  # Gaining experience pool
                self.experience_size += 1
                if self.experience_size < self.minibatch_size:  # Until it satisfy minibatch size
                    continue
                else:  # sample minibatch samples in experience pool
                    minibatch = np.random.choice(self.experience_pool, self.minibatch_size)
                minibatch = self._convert(minibatch)
                if terminal:
                    reward = instant_reward
                else:
                    target = self.sess.run(self.target, feed_dict={self.ipt: next_state})
                    reward = instant_reward + self.gamma * target.max()
                self.sess.run(
                    self.q.optimizer,
                    feed_dict={
                        self.ipt: minibatch.state_batch,
                        self.reward: minibatch.reward_batch,
                        self.opt: minibatch.action_batch,
                    }
                )
                if step % self.target_update_episode == 0:
                    self.q.update_target()

    def epsilon_greedy(self, epsilon, state):
        if epsilon < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.sess.run(self.q, feed_dict={self.ipt: state}).argmax()

    @property
    def action(self):
        return self._action

    @staticmethod
    def observe(game):
        return game.state, game.instant_reward, game.death

    @property
    def target(self):
        return self.q_networks['target']

    @property
    def q(self):
        return self.q_networks['approximation']

    @property
    def ipt(self):
        return self.q_network.ipt

    @property
    def reward(self):
        return self.q_network.reward

    @property
    def mast(self):
        return self.q_network.opt_mask

    def __getattr__(self, name):
        if name in self.hyper_params:
            return self.hyper_params[name]
        else:
            raise AttributeError()

    def __del__(self):
        self.sess.close()


if __name__ == '__main__':
    dqn = DQN(ipt_size=20, out_size=4)
