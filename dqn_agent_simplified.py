import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optimizer
import lunar_lander as lander
from collections import deque
import gym
import random
torch.device("cpu")
model = nn.Sequential(
    nn.Linear(8, 150),
    nn.ReLU(),
    nn.Linear(150, 120),
    nn.ReLU(),
    nn.Linear(120, 4),
)
loss_func = nn.MSELoss()
learning_rate = 0.001
optimizer = optimizer.Adam(model.parameters(), lr=learning_rate)

class DQNAgent:
    def __init__(self):
        self.epsilon = 1
        self.gamma = .99
        self.batch_size = 64
        self.learning_rate = 0.001
        self.memory = deque(maxlen=1000000)
        self.min_eps = 0.01
        self.model = model

    def store_experience(self, state, action, next_state, reward, finished):
        self.memory.append((state, action, next_state, reward, finished))

    def take_action(self, input):
        if np.random.random() <= self.epsilon:
            return np.random.choice(4)
        input = torch.Tensor(input)
        action_values = self.model(input)
        return torch.argmax(action_values).item()

    def replay_experiences(self):
        if len(self.memory) >= self.batch_size:
            sample_choices = np.array(self.memory)
            mini_batch_index =np.random.choice(len(sample_choices), self.batch_size)
            batch = random.sample(self.memory, self.batch_size)
            states = []
            actions =[]
            next_states = []
            rewards = []
            finishes = []

            for index in batch:
                states.append(index[0])
                actions.append(index[1])
                next_states.append(index[2])
                rewards.append(index[3])
                finishes.append(index[4])

            states = np.array(states)
            actions = np.array(actions)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
            finishes = np.array([index[4] for index in batch])
            #states = np.squeeze(states)
            #next_states = np.squeeze(next_states)
            #convert rewards and finishes to tensors
            rewards = torch.Tensor(rewards)
            finishes = torch.Tensor(finishes)
            states = torch.Tensor(states)
            next_states = torch.Tensor(next_states)

            q_vals_curr_state = self.model(states)
            q_vals_next_state = self.model(next_states)
            q_vals_target = q_vals_curr_state.clone()
            #print(q_vals_next_state.shape)
            max_q_values_next_state = torch.max(q_vals_next_state, dim=1)[0]
            q_vals_target[np.arange(self.batch_size), actions] = rewards + self.gamma * (max_q_values_next_state) * (1-finishes)
            loss = loss_func(q_vals_curr_state, q_vals_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.epsilon > self.min_eps:
                self.epsilon *= 0.996



if __name__ == '__main__':
    env = lander.LunarLander()
    #env.seed(0)
    agent = DQNAgent()
    num_episodes = 400
    np.random.seed(0)
    for i in range(num_episodes):
        score = 0
        state = env.reset()
        #state = np.reshape(state, (1, 8))
        finished = False
        for j in range(3000):
            action = agent.take_action(state)
            env.render()
            next_state, reward, finished, metadata = env.step(action)

            #next_state = np.reshape(next_state, (1, 8))
            agent.store_experience(state, action, next_state, reward, finished)
            score += reward
            state = next_state
            agent.replay_experiences()
            if finished:
                print("Episode = {}, Score = {}".format(i, score))
                break

