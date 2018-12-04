import tensorflow as tf
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt


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
        self.ipt = tf.placeholder(tf.float32, shape=(None, *self.ipt_shape, self.ipt_channel))
        self.reward = tf.placeholder(tf.float32, shape=(None, 1))

        # Below is the output mask that only the chosen neural (action) will output Q.
        self.opt_mask = tf.placeholder(tf.float32, shape=(self.opt_size, None))
        self.reg_lambda = 0.03
        self.alpha = 0.3

        c_strides = (1, 2, 2, 1)
        p_strides = (1, 2, 2, 1)
        k_size = (1, 3, 3, 1)
        stddev = 5e-2
        biases = 0.1
        model_name = ['Q', 'target']
        self.networks = {}
        for name in model_name:
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
                # self.PoolLayers(
                #     name=name,
                #     layer=2,
                #     ksize=k_size,
                #     strides=p_strides,
                # ),
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
                # self.PoolLayers(
                #     name=name,
                #     layer=4,
                #     ksize=k_size,
                #     strides=p_strides,
                # ),
                self.ConvLayers(
                    name=name,
                    layer=5,
                    kernel=(1, 1),
                    strides=c_strides,
                    number=128,
                    channels=32,
                    stddev=stddev,
                    bias=biases,
                ),
                self.ConvLayers(
                    name=name,
                    layer=6,
                    kernel=(1, 1),
                    strides=c_strides,
                    number=self.opt_size,
                    channels=128,
                    stddev=stddev,
                    bias=biases,
                )
                # self.FCLayers(
                #     name=name,
                #     layer=5,
                #     shape=128,
                #     stddev=stddev,
                #     bias=biases,
                #     regularizer=True,
                #     regularizer_weight=self.reg_lambda,
                #     activation=tf.nn.relu,
                # ),
                # self.FCLayers(
                #     name=name,
                #     layer=6,
                #     shape=self.opt_size,
                #     stddev=stddev,
                #     bias=biases,
                #     regularizer=None,
                #     regularizer_weight=self.reg_lambda,
                #     activation=None,
                # )
            ], name=name)
        self._action = 0
        self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
        self.saver = tf.train.Saver()

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
        if name == 'Q':
            current = tf.layers.Flatten()(current)
            return tf.matmul(current, self.opt_mask)
        else:
            return current

    @property
    def loss(self):
        return tf.reduce_mean(tf.square(self.reward - self['Q']))

    @staticmethod
    def _copy_model(sess):
        m1 = [t for t in tf.trainable_variables() if t.name.startswith('Q')]
        m1 = sorted(m1, key=lambda v: v.name)
        m2 = [t for t in tf.trainable_variables() if t.name.startswith('target')]
        m2 = sorted(m2, key=lambda v: v.name)

        ops = []
        for t1, t2 in zip(m1, m2):
            ops.append(t2.assign(t1))
        sess.run(ops)

    def __getitem__(self, item):
        return self.networks[item]


class DQN:

    exp = namedtuple('exp', ('state', 'action', 'instant', 'next_state', 'terminal'))

    def __init__(self, game):
        self.game = game
        self.ipt_size, self.opt_size = self.game.size
        self.q_network = QApproximation(self.ipt_size, self.opt_size, batch_size=128)
        self.global_step = tf.Variable(0, trainable=False)
        self.epsilon_decay = 0.9
        self.epsilon_base = 0.9
        self.epsilon_span = 100
        self._epsilon = tf.train.exponential_decay(
            self.epsilon_base,
            self.global_step,
            self.epsilon_span,
            self.epsilon_decay,
            staircase=True
        )
        self.actions = list(range(self.opt_size))
        self.experience_size = 0
        self.experience_pool = []
        self.episodes = 2000
        self.minibatch_size = 128
        self.target_update_episode = 10
        self.save_episode = 100
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.model_file = '../models/model.ckpt'
        self.hyper_params = {
            'gamma': 0.9,
        }

    # def gain_experiences(self, game):
    #     for _ in range(self.experience_size):
    #         state = self.observe(game)
    #         action_index = np.random.choice(self.actions)
    #         game.interact(action_index)
    #         reward = self.q.reward(game)
    #         next_state = self.observe(game)
    #         self.experience_pool.append(self.exp(
    #             state=state,
    #             action=action_index,
    #             instant=reward,
    #             next_state=next_state,
    #         ))
    #         if reward == -10:
    #             game.reset()

    @property
    def epsilon(self):
        return self.sess.run(self._epsilon)

    def _convert(self, minibatch):
        'Convert minibatch from namedtuple to multi-dimensional matrix'
        batch = {
            'state': [],
            'reward': [],
            'action': [],
        }
        for block in minibatch:
            batch['state'].append(block.state)
            batch['action'].append([1. if _ == block.action else 0. for _ in self.actions])
            if block.terminal:
                reward = block.instant
            else:
                target = self.sess.run(self.target, feed_dict={self.ipt: np.array([block.next_state])})
                reward = block.instant + self.gamma * target.max()
            batch['reward'].append(reward)
        batch['action'] = np.array(batch['action']).T
        batch['reward'] = np.array([batch['reward']]).T
        return batch

    def train(self, window=None):
        try:
            self.q_network.saver.restore(self.sess, self.model_file)
        except ValueError:
            print('First-time train')
        except tf.errors.InvalidArgumentError:
            print('New game')
        except tf.errors.DataLossError:
            print('FATAL ERROR, start new game')
        self.game.reset()
        # plt.figure('Loss')
        # plt.ion()
        episodes = []
        lossarr = []
        lossave = []
        for episode in range(self.episodes):
            state = self.game.state
            while True:
                state = state.reshape((self.ipt_size, self.ipt_size, 1), order='F')
                epsilon = np.random.rand()
                action = self.epsilon_greedy(epsilon, state)
                next_state, reward, terminal, _ = self.game.step(action)
                next_state = next_state.reshape((self.ipt_size, self.ipt_size, 1), order='F')
                if self.game.eat:
                    self.game.new_food()
                self.experience_pool.append(self.exp(
                    state=state,
                    action=action,
                    instant=reward,
                    next_state=next_state,
                    terminal=terminal,
                ))  # Gaining experience pool
                self.game.render(window)
                if terminal:
                    if self.experience_size >= self.minibatch_size:  # Until it satisfy minibatch size
                        choices = np.random.choice(list(range(self.experience_size)), self.minibatch_size)
                        minibatch = [self.experience_pool[_] for _ in choices]
                        episodes.append(episode)
                        self.sess.run(tf.assign(self.global_step, episode))
                        batch = self._convert(minibatch)
                        _, loss = self.sess.run(
                            [self.optimizer, self.q_network.loss],
                            feed_dict={
                                self.ipt: batch['state'],
                                self.mask: batch['action'],
                                self.reward: batch['reward'],
                            }
                        )
                        lossarr.append(loss)
                        lossave.append(sum(lossarr)/(episode + 1))
                        #self.show(episodes, lossarr, lossave)
                        if episode % self.target_update_episode == 0:
                            self.q_network._copy_model(self.sess)
                        if episode % self.save_episode == 0:
                            self.q_network.saver.save(self.sess, self.model_file)
                    self.game.reset()
                    break
                self.experience_size += 1
        # plt.ioff()
        # plt.show()
        self.game.close(window)

    def show(self, episodes, lossarr, lossave):
        plt.cla()
        plt.title('Interactive loss over episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.plot(episodes, lossarr, 'm+-', label='Instant loss')
        plt.plot(episodes, lossave, 'co-', label='Moving Average loss')
        plt.legend()
        plt.pause(0.01)

    def epsilon_greedy(self, epsilon, state):
        if epsilon < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.sess.run(self.q, feed_dict={self.ipt: [state], self.mask: np.array([1., 1., 1., 1.]).reshape(4, 1)}).argmax()

    @property
    def action(self):
        return self._action

    @property
    def target(self):
        return self.q_network['target']

    @property
    def q(self):
        return self.q_network['Q']

    @property
    def ipt(self):
        return self.q_network.ipt

    @property
    def reward(self):
        return self.q_network.reward

    @property
    def mask(self):
        return self.q_network.opt_mask

    @property
    def optimizer(self):
        return self.q_network.optimizer

    def __getattr__(self, name):
        if name in self.hyper_params:
            return self.hyper_params[name]
        else:
            raise AttributeError()

    def __del__(self):
        self.sess.close()


if __name__ == '__main__':
    from snake import Game
    from window import Window
    number = 6
    block_size = 20
    g = Game(number=number)
    window = Window(number=number, block_size=block_size, expansion=1.5, speed=0.2)
    dqn = DQN(game=g)
    dqn.train(window=window)
