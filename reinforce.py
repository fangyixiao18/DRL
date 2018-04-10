import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
import random
import copy

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Reinforce(object):
    
    # Implementation of the policy gradient method REINFORCE.
    def __init__(self, env, lr):
        self.env = env
        self.states_dim = self.env.observation_space.shape[0]
        self.actions_dim = self.env.action_space.n
        self.lr = lr
        self.gamma = 1.0
        self.build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

    def build_net(self):
        # input
        with tf.variable_scope('input'):
            self.input = tf.placeholder(tf.float32, [None, self.states_dim])

        # layer 1
        with tf.variable_scope('layer1'):
            layer1 = tf.layers.dense(self.input, 16, 
                                    kernel_initializer = tf.random_normal_initializer(), 
                                    activation = tf.nn.relu)
            self.checkl1 = tf.norm(layer1) / 4
        
        # layer 2
        with tf.variable_scope('layer2'):
            layer2 = tf.layers.dense(layer1, 16, 
                                    kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.5), 
                                    activation = tf.nn.relu)
            self.checkl2 = tf.norm(layer2) / 4

        # layer 3
        with tf.variable_scope('layer3'):
            layer3 = tf.layers.dense(layer2, 16, 
                                    kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.2), 
                                    activation = tf.nn.relu)
            self.checkl3 = tf.norm(layer3) / 4

        # output layer
        with tf.variable_scope('outputlayer'):
            self.output = tf.layers.dense(layer3, self.actions_dim)
            self.actions_pred = tf.nn.softmax(self.output)

        # loss function
        with tf.variable_scope('loss'):
            # variables used in loss
            self.Gt = tf.placeholder(tf.float32, name = 'return') # the sum of the return reawards
            self.actions_cast = tf.placeholder(tf.float32, name = 'actions_cast') # the one hot actions
            # loss
            self.J = tf.log(tf.reduce_sum(tf.multiply(self.actions_pred, self.actions_cast), axis = 1))
            self.loss = -tf.reduce_mean(tf.multiply(self.Gt, self.J))

        # train operation
        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)



    def train(self, gamma = 1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        # generate the dataset
        states, actions, rewards = self.generate_episode()

        # start training
        steps = len(states)
        # for i in range(steps):
            # find action a_t at state s_t
            # actions_pred = self.sess.run(self.actions_pred, 
            #                             feed_dict = {self.input: states[i]})
            # selected_act = np.argmax(actions_pred)
            # print(actions_pred)
            
            # training
        Gt = np.zeros(steps)
        t = steps - 1
        while(t >= 0):
            temp = 0
            k = copy.copy(t)
            while(k < steps):
                temp += gamma**(k - t) * rewards[k]
                k += 1
            Gt[t] = copy.copy(temp)*0.01
            t = t - 1
        # print(Gt)
        # print(np.array(states).shape)
        
        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict = {self.input: states,
                                self.Gt: Gt,
                                self.actions_cast: actions})
        # GtJ = self.sess.run(self.J, 
        #                     feed_dict = {self.input: states[:][0][:],
        #                     self.Gt: rewards,
        #                     self.actions_cast: actions})

        # actionpred = self.sess.run(self.actions_pred, 
        #                         feed_dict = {self.input: states[:][0][:]})
        # print(actionpred)
        # print(loss)
        # print(GtJ)
        # print(rewards)
        # checkl1, checkl2, checkl3 = self.sess.run([self.checkl1, self.checkl2, self.checkl3],
        #     feed_dict = {self.input: states[:][0][:]})
        # print(checkl1, checkl2, checkl3)

        # return


    def act(self, actions):
        random_num = random.random()
        if random_num < actions[0, 0]:
            action = 0
        elif actions[0, 0] <= random_num < (actions[0, 0] + actions[0, 1]):
            action = 1
        elif (actions[0, 0] + actions[0, 1]) <= random_num < (actions[0, 0] + actions[0, 1] + actions[0, 2]):
            action = 2
        elif (actions[0, 0] + actions[0, 1] + actions[0, 2]) <= random_num < 1:
            action = 3
        return action


    def generate_episode(self, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        # initialization
        done = False
        state = self.env.reset()
        while True:
            if render:
                self.env.render()
            # record states
            states.append(state)

            # obtain action in certain state
            actions_pred = self.sess.run(self.actions_pred, 
                                    feed_dict = {self.input: [state]})
            action = self.act(actions_pred)
            action_onehot = keras.utils.np_utils.to_categorical(action, 4)

            # record actions
            actions.append(action_onehot)

            # take actions and obtain other info
            next_state, reward, done, _ = self.env.step(action)

            # record rewards
            rewards.append(reward)
            
            # update the state
            state = next_state

            # episode ends
            if done:
                break

        return states, actions, rewards


    def test_model(self, render = False):
        # initialization
        episodes = 100
        rewards_sum = [0] * episodes

        # test 100 episodes
        for i in range(episodes):
            state = self.env.reset()
            # print(i)
            while True:
                # whether redner or not
                if render:
                    self.env.render()

                # obtain action in certain state
                actions_pred = self.sess.run(self.actions_pred, 
                                        feed_dict = {self.input: [state]})
                action = self.act(actions_pred)

                # take actions and obtain other info
                next_state, reward, done, _ = self.env.step(action)
                rewards_sum[i] += reward

                # update the state
                state = next_state

                # episode ends
                if done:
                    break

        # calculate the mean and std
        reward_avg = np.mean(rewards_sum)
        reward_std = np.std(rewards_sum)

        return reward_avg, reward_std


    def plot_figure(self, x, y, yerr, x_name, y_name):
        plt.figure()
        plt.plot(x,y)
        plt.errorbar(x, y, yerr, capsize = 3)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.savefig('10703reinforce.png')
        plt.show()
        


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the policy model from file.
    # with open(model_config_path, 'r') as f:
    #     model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.

    # settings
    x = []
    y = []
    
    # learning
    agent = Reinforce(env, lr)
    for i in range(num_episodes):
        agent.train()

        # test model
        if i % 500 == 0:
            print('Test model at %d episode' %i)
            reward_avg, reward_std = agent.test_model()
            print('reward_avg:', reward_avg, '\n')
            x.append(i)
            y.append(reward_avg)

    # draw the figure of rewards
    agent.plot_figure(x, y, reward_std, 'episodes', 'average_rewards')


if __name__ == '__main__':
    main(sys.argv)
