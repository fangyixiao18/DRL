# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 19:29:56 2018

@author: fangyx
"""

import lake_envs as lake_env
import numpy as np
import gym
import time
import rl
#import matplotlib.pyplot as plt

    
''' initialize the environment '''
start = time.time()

# select the map and make the environment
selected_env = 'Deterministic-8x8-FrozenLake-v0'

env = gym.make(selected_env)
print('Environment has %d states and %d actions.' % (env.nS, env.nA))
env.render()

# obtain map edge length
mapedge = int(np.sqrt(env.nS))



''' learning '''
# initialize the discount factor
gamma = 0.9
# policy iterations
optimal_policy, optimal_value_func, nPi, nVi = rl.policy_iteration_sync(
        env, gamma, max_iterations=int(1e3), tol=1e-3)

# run the optimal policy
total_reward = rl.run_policy_deterministic(env, optimal_policy, gamma)
print(total_reward)

optimal_policy = np.reshape(optimal_policy, (mapedge, mapedge))
print(optimal_policy,"\n")


end = time.time()
timeconsuming = 1000 * (end - start)



''' the result '''
print("Number of Pollicy Improvement Iterations:", nPi)
print("Total number of Pollicy Evaluation Iterations:", nVi)
print("Time :", timeconsuming)
print(np.reshape(optimal_value_func, (mapedge, mapedge)))

#draw the figure
rl.draw(optimal_value_func, mapedge)


