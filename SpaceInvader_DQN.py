# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 16:16:01 2018

@author: fangyx
"""

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import cv2
import random
from collections import deque
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#np.set_printoptions(threshold = np.nan)

# hyperparameters for Q-learning
env_name = 'SpaceInvaders-v0'
gamma = 0.99
init_epsilon = 0.5
final_epsilon = 0.05
memory_size = 1000000
batch_size = 32
burn_in = 10000
prob = 0.5

# hyperparameters for iterations
episodes = 10000
test = 1


class SI_DQN:
    def __init__(self, env, sess):
        # obtain the information of the environment
        self.env = env
        self.sess = sess
        self.epsilon = init_epsilon
        self.memory = deque()
        self.num_actions = env.action_space.n # number of the actions
#        self.num_ob = env.observation_space.shape[0] # number of the observations
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        
        
    def build_network(self, learning_rate = 0.001):
        # CNN frame
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        
        def bias_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
        def max_pool_2x2(x, size):
            return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                                  strides=[1, size, size, 1], padding='SAME')
        
        # input image
        self.input_image = tf.placeholder(tf.float32, [None, 84, 84, 1], name = 'input_image')
        
        # layer1 setting
        W1 = weight_variable([10, 10, 1, 16])
        b1 = bias_variable([16])
        conv1 = tf.nn.relu(conv2d(self.input_image, W1) + b1)
        pool1 = max_pool_2x2(conv1, 2)
        
        # layer2 setting
        W2 = weight_variable([5, 5, 16, 32])
        b2 = bias_variable([32])
        conv2 = tf.nn.relu(conv2d(pool1, W2) + b2)
        pool2 = max_pool_2x2(conv2, 6)
        
#        # layer3 setting
#        W3 = weight_variable([5, 5, 32, 64])
#        b3 = bias_variable([64])
#        conv3 = tf.nn.relu(conv2d(pool2, W3, 2) + b3)
#        pool3 = max_pool_2x2(conv2, 3)
        
        # fully connected layer
        input_fc = tf.reshape(pool2, [-1, 7*7*32])
        W_fc = weight_variable([7*7*32, 1024])
        b_fc = bias_variable([1024])
        output_fc = tf.nn.relu(tf.matmul(input_fc, W_fc) + b_fc)
        
        # dropout step
        output_dp = tf.nn.dropout(output_fc, keep_prob=prob)
        # output layer
        in_size = int(output_dp.get_shape()[1])
        W_out = weight_variable([in_size, self.num_actions])
        b_out = bias_variable([self.num_actions])
        self.Qpred = tf.nn.relu(tf.matmul(output_dp, W_out) + b_out)
        # the target Q-values
        self.Qtarget = tf.placeholder(tf.float32, [None, None], name = 'Qtarget')
        # one hot presentation
        self.act = tf.placeholder(tf.float32, [None, self.num_actions], name = 'onehot')
        self.Q_action = tf.reduce_sum(tf.multiply(self.Qpred, self.act), reduction_indices = 1)
        ##################
        self.Q_action = tf.reshape(self.Q_action, [batch_size, 1])
        ##################
        
        # the training section
        self.loss = tf.reduce_mean(tf.square(self.Qtarget - self.Q_action))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        
    
    
    def train_network(self):
        minibatch = random.sample(self.memory, batch_size)
        state = [info[0] for info in minibatch]
        action = [info[1] for info in minibatch]
        reward = [info[2] for info in minibatch]
        next_state = [info[3] for info in minibatch]
        
        action = np.matrix(action)
        
        Qvalue = self.sess.run(self.Qpred, feed_dict={self.input_image: next_state})
        Qtar = []
        for i in range(0, batch_size):
            done = minibatch[i][4]
            if done:
                Qtar.append(reward[i])
            else:
                Qtar.append(reward[i] + gamma * np.max(Qvalue[i, :]))
        
        _, losses = self.sess.run([self.train_op, self.loss], 
                                  feed_dict = {self.Qtarget: np.matrix(Qtar),
                                  self.act: action,
                                  self.input_image: state})
    
    
    def remember(self, state, action, reward, next_state, done):
        # convert to one hot action
        one_hot_action = np.zeros(self.num_actions)
        one_hot_action[action] = 1
        
        # deal with the state
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (84, 84))
        state = state.reshape((84, 84, 1)) 
        
        # deal with the next state
        next_state = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
        next_state = cv2.resize(next_state, (84, 84))
        next_state = next_state.reshape((84, 84, 1)) 
        
        # add to the memory
        self.memory.append((state, one_hot_action, reward, next_state, done))
        
        if len(self.memory) > memory_size:
            self.memory.popleft()
    
    
    def epsilon_greedy_policy(self, current_state):
        # Creating epsilon greedy probabilities to sample from.
        current_state = cv2.cvtColor(current_state, cv2.COLOR_RGB2GRAY)
        current_state = cv2.resize(current_state, (84, 84))
        current_state = current_state.reshape((1, 84, 84, 1))          
        Q_value = self.sess.run(self.Qpred, feed_dict = {self.input_image: current_state})
        if self.epsilon > 0.05:
            self.epsilon -= (init_epsilon - final_epsilon)/100000
        
        if random.random() <= self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(Q_value)
    
    
    def test_policy(self, current_state, test_epsilon = 0.05):
        # epsilon = 0.05
        current_state = cv2.cvtColor(current_state, cv2.COLOR_RGB2GRAY)
        current_state = cv2.resize(current_state, (84, 84))
        current_state = current_state.reshape((1, 84, 84, 1))
        Q_value = self.sess.run(self.Qpred, feed_dict = {self.input_image: current_state})
        if random.random() <= test_epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(Q_value)
    
    
    def test(self):
        state = self.env.reset()
        reward_sum = 0
        counter = 0
        while True:
            self.env.render()
            counter += 1
            act = self.test_policy(state)
            next_s, reward, done, _ = self.env.step(act)
            reward_sum += reward
            state = next_s
            if done:
                break
        return reward_sum
    
#------------------------------------------------------------------------------
def main():
    env = gym.make(env_name)
    
    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)

    # the main part
    agent = SI_DQN(env, sess)
    StartTraining = True
    
    for e in range(episodes):
        # initialize
        state = env.reset()
        while True:
            # get the action
            action = agent.epsilon_greedy_policy(state)
            # take the action and obtain the info
            next_s, reward, done, _ = env.step(action)
            # append memomry
            agent.remember(state, action, reward, next_s, done)
            if len(agent.memory) > burn_in:
                if StartTraining:
                    print('Start Training!')
                    StartTraining = False
                # train the network
                print(e)
                agent.train_network()

            # update state
            state = next_s
            if done:
#                print(counter)
                break
            
        # test
        if e % 5 == 0:
            reward_test = 0
            for i in range(test):
                reward_test += agent.test()
            avg_reward = reward_test  / test
            print('Episodes:', e, '  The average reward:', avg_reward)

        
if __name__ == '__main__':
    main()