# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 19:08:55 2018

@author: Yixiao Fang(Andrew ID: yixiaof)
"""
# CP 10x18x8

import keras
import tensorflow as tf
import numpy as np
import gym
import sys
import argparse
import random
import matplotlib.pyplot as plt
from collections import deque
import time

# hyperparameters for Q-learning
env_name = 'MountainCar-v0'
gamma = 0.99
init_epsilon = 0.3
final_epsilon = 0.05
memory_size = 50000
batch_size = 32
burn_in = 10000

# hyperparameters for iterations
episodes = 5000
test = 20
limit = 1000

class Dueling_DQN(): 
    def __init__(self, env, sess):
        # obtain the information of the environment
        self.env = env
        self.sess = sess
        self.epsilon = init_epsilon
        self.memory = deque()
        self.num_actions = env.action_space.n # number of the actions
        self.num_ob = env.observation_space.shape[0] # number of the observations
        self.build_network()
        self.sess.run(tf.global_variables_initializer())

        
    # build the neural network    
    def build_network(self, learning_rate = 0.0001):
        # deep nueral network
        nodes1 = 10
        nodes2 = 18
        nodes3 = 8
        
        # the frame of the nueral network
        self.x = tf.placeholder(tf.float32, [None, self.num_ob]) # input - state features
        layer1 = tf.layers.dense(self.x, nodes1, 
                                 kernel_initializer = tf.truncated_normal_initializer(stddev=0.1), 
                                 activation = tf.nn.relu)
        layer2 = tf.layers.dense(layer1, nodes2, 
                                 kernel_initializer = tf.truncated_normal_initializer(stddev=0.1), 
                                 activation = tf.nn.relu)
        layer3_V = tf.layers.dense(layer2, nodes3, 
                                 kernel_initializer = tf.truncated_normal_initializer(stddev=0.1), 
                                 activation = tf.nn.relu)
        # for dueling DQN
        layer3_A = tf.layers.dense(layer2, nodes3, 
                                 kernel_initializer = tf.truncated_normal_initializer(stddev=0.1), 
                                 activation = tf.nn.relu)
        
        # two dueling stream
        self.V = tf.layers.dense(layer3_V, 1)
        self.A = tf.layers.dense(layer3_A, self.num_actions)
        
        meannum = tf.reduce_mean(self.A)
        self.Qpred = self.V + (self.A - meannum)
        
        # the calculation of the tensor
        self.Qtarget = tf.placeholder(tf.float32, [None, None]) # input - the target Q-values
        self.act = tf.placeholder(tf.float32, [None, self.num_actions]) # input -  one hot presentation
        self.Q_action = tf.reduce_sum(tf.multiply(self.Qpred, self.act), reduction_indices = 1)

        
        # loss function and the optimiazing
        self.loss = tf.reduce_mean(tf.square(self.Qtarget - self.Q_action))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        

    def train_network(self):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        
        # extract the information from the minibatch
        minibatch = random.sample(self.memory, batch_size)
        dim = len(minibatch)
        state = np.zeros((dim, self.num_ob))
        action = np.zeros((dim, self.num_actions))
        reward = np.zeros(dim)
        next_state = np.zeros((dim, self.num_ob))
        for i in range(dim):
            state[i] = minibatch[i][0]
            action[i] = minibatch[i][1]
            reward[i] = minibatch[i][2]
            next_state[i] = minibatch[i][3]
        
        Q = self.sess.run(self.Qpred, feed_dict={self.x: next_state})
        
        Qtar = []
        for i in range(0, batch_size):
            done = minibatch[i][4]
            if done:
                Qtar.append(reward[i])
            else:
                Qtar.append(reward[i] + gamma * np.max(Q[i, :]))
        
        _, losses = self.sess.run([self.train_op, self.loss], 
                                  feed_dict = {self.Qtarget: np.matrix(Qtar),
                                  self.act: action,
                                  self.x: state})

    
    def remember(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.num_actions)
        one_hot_action[action] = 1
        
        self.memory.append((state, one_hot_action, reward, next_state, done))
        
        if len(self.memory) > memory_size:
            self.memory.popleft()
        
    
    
    def epsilon_greedy_policy(self, current_state):
        # Creating epsilon greedy probabilities to sample from. 
        current_state = np.matrix(current_state)           
        Q_value = self.sess.run(self.Qpred, feed_dict = {self.x: current_state})
        if self.epsilon > 0.05:
            self.epsilon -= (init_epsilon - final_epsilon)/100000
        
        if random.random() <= self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(Q_value)
        

    def test_policy(self, current_state, test_epsilon = 0.05):
        # epsilon = 0.05
        Q_value = self.sess.run(self.Qpred, feed_dict = {self.x: current_state})
        if random.random() <= test_epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(Q_value)    
        

    def test(self, showv):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        state = self.env.reset()
        reward_sum = 0
        counter = 0
        while True:
            if showv:
                self.env.render()
            counter += 1
            state = np.matrix(state)
            act = self.test_policy(state)
            next_s, reward, done, _ = self.env.step(act)
            reward_sum += reward
            state = next_s
            if done or counter >= limit:
                break
        return reward_sum
    
    def plot_figure(self, x, y):
        plt.figure()
        plt.plot(x,y)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()
        plt.savefig('./test')
    
# -----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    return parser.parse_args()


def main(args):

    args = parse_arguments()
    environment_name = args.env
    env = gym.make(env_name)
    env = env.unwrapped
    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session. 
    keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train / test it. 
    # initialize OpenAI Gym env and dqn agent
    agent = Dueling_DQN(env, sess)
    TrainingStart = True
    testmodel = True
    x = []
    y = []
    epi = []
    trainreward = []
    rsum = 0
    for e in range(episodes):
        # initialize
        state = env.reset()
        counter = 0
        while True:
            counter += 1
            # get the action
            action = agent.epsilon_greedy_policy(state)
            # take the action and obtain the info
            next_s, reward, done, _ = env.step(action)
            rsum += reward
            # append memomry
            agent.remember(state, action, reward, next_s, done)
            if len(agent.memory) > burn_in:
                if TrainingStart:
                    print('Start Training!')
                    TrainingStart = False
                # train the network
                agent.train_network()

            # update state
            state = next_s
            if done or counter >= limit:
#                print(counter)
                break
        
        if e % 20 == 0 and e > 0:
            epi.append(e)
            trainreward.append(rsum/20)
            rsum = 0
            
        # test
        if e % 50 == 0:
            reward_test = 0
            for i in range(test):
                reward_test += agent.test(False)
            avg_reward = reward_test  / test
            x.append(e)
            y.append(avg_reward)
            print('Episodes:', e, '  The average reward:', avg_reward)
            
            
        if e > 3000 and testmodel and avg_reward > -170:
            rwd = np.zeros(100)
            for i in range(100):
                rwd[i] = agent.test(False)
            rwd_mean = np.mean(rwd)
            rwd_std = np.std(rwd)
            testmodel = False
            
            
        if e % 1666 == 0:
            print('Capture Video!!!')
            print('Capture Video!!!')
            print('Capture Video!!!')
            print('Capture Video!!!')
            print('Capture Video!!!')
            time.sleep(8)
            rr = agent.test(False)
    
    # plot the reward
    agent.plot_figure(x, y)
    print('Mean: ', rwd_mean)
    print('Std: ', rwd_std)
        
    agent.plot_figure(epi, trainreward)
    
if __name__ == '__main__':
    main(sys.argv)
    

