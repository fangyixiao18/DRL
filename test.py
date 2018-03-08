# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 19:08:55 2018

@author: fangyx
"""

import gym

env = gym.make('CartPole-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())