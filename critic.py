import tensorflow as tf
import numpy as np

class Critic:
    def __init__(self, batch_size,
                 learning_rate = 0.001, state_size = 27, 
                 action_size = 4, name='value_function', 
                 l2_loss_factor = 0.0, fc_sizes = [32, 64, 32]):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size
        self.l2_loss_factor = l2_loss_factor
        self.fc_sizes = fc_sizes
        with tf.variable_scope(name):
            self.state_input = tf.placeholder(tf.float32, [None, self.state_size], name='state_input')
            self.input_dr = tf.placeholder(tf.float32, [None, 1], name='state_input')
            
            self.fc1 = tf.contrib.layers.fully_connected(self.state_input, self.fc_sizes[0])
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, self.fc_sizes[1])
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, self.fc_sizes[2])
            
            self.v = tf.contrib.layers.fully_connected(self.fc2, 1)
            self.adv = self.input_dr - self.v
            
            self.loss = tf.reduce_mean(tf.square(self.adv))
            self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)