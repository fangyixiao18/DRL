# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 19:29:56 2018

@author: fangyx
"""

#import lake_envs as lake_env
import numpy as np
import gym
import time
import rl
#import matplotlib.pyplot as plt



''' async_custom'''
''' initialize the environment '''
start = time.time()

# select the map and make the environment
selected_env = 'Stochastic-8x8-FrozenLake-v0'

env = gym.make(selected_env)
print('Environment has %d states and %d actions.' % (env.nS, env.nA))

# obtain map edge length
mapedge = int(np.sqrt(env.nS))



''' learning '''
# initialize the discount factor
gamma = 0.9
# policy iterations
optimal_value_func, nVi = rl.value_iteration_async_custom(
        env, gamma, max_iterations=int(1e3), tol=1e-3)

ov = np.reshape(optimal_value_func, (mapedge, mapedge))

end = time.time()
timeconsuming = 1000 * (end - start)



''' the result '''
#print("Number of Pollicy Improvement iterations:", nPi)
print("Total number of Value Iterations:", nVi)
print("Time :", timeconsuming)
#print(optimal_value_func)


''' draw the figure '''
rl.draw(optimal_value_func, mapedge)

