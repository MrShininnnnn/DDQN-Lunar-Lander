#! /usr/bin/env python
__author__ = 'Ning Shi'
__email__ = 'mrshininnnnn@gmail.com'

import os

class Config():
    # configuration
    def __init__(self):
        # path
        self.CURR_PATH = os.path.abspath('')
        self.OUTPUT_PATH = os.path.join(self.CURR_PATH, 'output')
        self.DDQN_AGENT_PATH = os.path.join(self.OUTPUT_PATH, 'ddqn_agent')
        self.DDQN_CHECKPOINT_PATH = os.path.join(self.DDQN_AGENT_PATH, 'policy_model_checkpoint.pth')
        self.DDQN_RESULT_IMG_PATH = os.path.join(self.DDQN_AGENT_PATH, 'result_img_{}.png')
        if not os.path.exists(self.DDQN_AGENT_PATH): 
            os.makedirs(p)
        # epsilon greedy search
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.004
        # train
        self.n_episodes = 1000
        self.batch_size = 64
        self.best_avg_rewards = 100
        self.rewards_window_size = 100
        # DDQN
        self.hidden_size = 64
        self.memory_size = 100000
        self.update_step = 4
        self.learning_rate = 5e-4
        self.gamma = 0.99
        self.tau = 1e-2
        # evaluation
        self.test_episodes = 100

