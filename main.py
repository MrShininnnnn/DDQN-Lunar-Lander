#! /usr/bin/env python
__author__ = 'Ning Shi'
__email__ = 'mrshininnnnn@gmail.com'


# import dependency
import gym
from gym import wrappers

import torch
import numpy as np
from tqdm import trange
from collections import deque
import matplotlib.pyplot as plt

from config import Config
from src.agent import DDQN_Agent


class LunarLander():
    """docstring for LunarLander"""
    def __init__(self):
        super(LunarLander, self).__init__()
        self.config = Config()
        self.env = gym.make('LunarLander-v2')
        self.n_states, self.n_actions = self.env.observation_space.shape[0], self.env.action_space.n
        self.initialize_epsilon()
        self.agent = DDQN_Agent(
            n_states=self.n_states, 
            n_actions=self.n_actions, 
            batch_size=self.config.batch_size, 
            hidden_size=self.config.hidden_size, 
            memory_size=self.config.memory_size, 
            update_step=self.config.update_step, 
            learning_rate=self.config.learning_rate, 
            gamma=self.config.gamma, 
            tau=self.config.tau
            )

    def initialize_epsilon(self):
        # initialize epsilon values for greedy search
        self.epsilon_array = np.zeros((self.config.n_episodes))
        for i in range(self.config.n_episodes):
            epsilon = self.config.min_epsilon + \
            (self.config.max_epsilon - self.config.min_epsilon) * np.exp(-self.config.decay_rate*i)
            self.epsilon_array[i] = epsilon

    def train(self):
        total_rewards = []
        rewards_deque = deque(maxlen=self.config.rewards_window_size)
        t = trange(self.config.n_episodes)
        for episode in t:
            # initialize the state
            cur_state = self.env.reset()
            done = False
            rewards = 0
            epsilon = self.epsilon_array[episode]
            while not done:
                action = self.agent.act(cur_state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(cur_state, action, reward, next_state, done)
                cur_state = next_state
                rewards += reward
            # update information
            total_rewards.append(rewards)
            rewards_deque.append(rewards)
            # average reward value given the reward window
            avg_rewards = np.mean(rewards_deque)
            t.set_description(
                'Episode {} Epsilon {:.2f} Reward {:.2f} Avg_Reward {:.2f} Best_Avg_Reward {:.2f}'.format(
                    episode + 1, epsilon, rewards, avg_rewards, self.config.best_avg_rewards))
            t.refresh()
            # evaluation
            if avg_rewards >= self.config.best_avg_rewards: 
                self.config.best_avg_rewards = avg_rewards
                torch.save(self.agent.policy_model.state_dict(), self.config.DDQN_CHECKPOINT_PATH)
            # the game is solved by earning more than +200 rewards for a single episode
            if self.config.best_avg_rewards > 200:
                self.env.close()
                # show rewards change in training
                plt.subplots(figsize = (5, 5), dpi=100)
                plt.plot(total_rewards)
                plt.ylabel('Total Reward', fontsize=12)
                plt.xlabel('Episode', fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.title('Total Rewards Per Episode for {} Training Episodes'.format(episode + 1), fontsize=12)
                plt.savefig(self.config.DDQN_RESULT_IMG_PATH.format(0), dpi=100, bbox_inches='tight')
                print('Save training rewards plot as {}.'.format(self.config.DDQN_RESULT_IMG_PATH.format(0)))
                break

    def eva(self):
        agent = DDQN_Agent(
            n_states=self.n_states, 
            n_actions=self.n_actions, 
            batch_size=self.config.batch_size, 
            hidden_size=self.config.hidden_size, 
            memory_size=self.config.memory_size, 
            update_step=self.config.update_step, 
            learning_rate=self.config.learning_rate, 
            gamma=self.config.gamma, 
            tau=self.config.tau
            )
        test_reward_array = np.zeros(100)
        # load check point to restore the model
        agent.policy_model.load_state_dict(
            torch.load(self.config.DDQN_CHECKPOINT_PATH, map_location=agent.device))
        t = trange(self.config.test_episodes)
        for episode in t:
            state = self.env.reset()
            done = False
            rewards = 0
            while not done:
                # disable epsilon greedy search
                action = agent.act(state, epsilon=0)
                state, reward, done, _ = self.env.step(action)
                rewards += reward
            t.set_description('Episode {:.2f} Reward {:.2f}'.format(episode + 1, rewards))
            t.refresh()
            test_reward_array[episode] = rewards
        self.env.close()
        # show the evaluation results
        avg_test_reward = round(np.mean(test_reward_array), 2)
        plt.subplots(figsize = (5, 5), dpi=100)
        plt.plot(test_reward_array)
        plt.ylabel('Total Reward', fontsize=12)
        plt.xlabel('Trial', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Total Rewards Per Trial for 100 Trials - Average: {:.2f}'.format(avg_test_reward), 
                  fontsize=12)
        plt.savefig(self.config.DDQN_RESULT_IMG_PATH.format(1), dpi=100, bbox_inches='tight')
        print('Save evaluation rewards plot as {}.'.format(self.config.DDQN_RESULT_IMG_PATH.format(1)))
        # play a round
        env = wrappers.Monitor(self.env, self.config.DDQN_AGENT_PATH, force=True)
        state = env.reset()
        done = False
        rewards = 0.
        while not done:
            # disable epsilon greedy search
            action = agent.act(state, epsilon=0)
            state, reward, done, _ = env.step(action)
            rewards += reward
        env.close()
        print('Total Rewards in a game: {:.2f}'.format(rewards))
        print('Save video record to {}.'.format(self.config.DDQN_AGENT_PATH))

def main():
    print('\nInitialize the environment...')
    ll = LunarLander()
    print('Training...')
    ll.train()
    print('Evaluating...')
    ll.eva()
    print('Done.')

if __name__ == '__main__':
      main()