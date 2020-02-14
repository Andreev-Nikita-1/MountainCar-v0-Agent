import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import copy
import random
import pylab
import numpy as np

env = gym.make("MountainCarContinuous-v0")
batch_size = 256
gamma = 0.998
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
outputs = 10


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, outputs)
        )
        self.bias = torch.tensor([0.52, 0.]).to(device).float()

    def action(self, x):
        return self(x).max(1)[1]

    def qmax(self, x):
        return self(x).max(1)[0]

    def forward(self, x):
        return self.model(x + self.bias)


def init_model():
    def init_weights(layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)

    model = Model()
    target_model = copy.deepcopy(model)
    model.apply(init_weights)
    model.train()
    model.to(device)
    target_model.train()
    target_model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00003)
    return model, target_model, optimizer


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return list(zip(*random.sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)


def fit(batch):
    state, action, reward, next_state, done = batch
    state = torch.tensor(state).to(device).float()
    next_state = torch.tensor(next_state).to(device).float()
    reward = torch.tensor(reward).to(device).float()
    action = torch.tensor(action).to(device).unsqueeze(1)
    with torch.no_grad():
        target_q = target_model.qmax(next_state).view(-1)
    target_q = reward + target_q * gamma
    q = model(state).gather(1, action).view(-1)
    loss_critic = F.smooth_l1_loss(q, target_q)
    optimizer.zero_grad()
    loss_critic.backward()
    optimizer.step()


def select_action(state, model, epsilon):
    if random.random() < epsilon:
        return np.random.randint(0, outputs)
    return model.action(torch.tensor(state).to(device).float().unsqueeze(0)).item()


def potential(state):
    return 400 * state[1] ** 2 + 30 * max(0, state[0] - 0.2)


def render(render=True):
    state = env.reset()
    total_reward = 0
    total_mod_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = select_action(state, target_model, 0)
        new_state, reward, done, _ = env.step([actions[action]])
        if reward > 0:
            print("victory")
        total_reward += reward
        total_mod_reward += reward + gamma * potential(new_state) - potential(state)
        state = new_state
    return total_reward, total_mod_reward


memory = Memory(10000)
rewards = []
modified_rewards = []
model, target_model, optimizer = init_model()
actions = np.arange(-2, 2 + 2 / (outputs - 1), 4 / (outputs - 1))


def teach(max_iter, update_period, render_period):
    state = env.reset()
    global target_model
    for i in range(1, max_iter):
        epsilon = 1 - np.exp(-i / max_iter / 2)
        with torch.no_grad():
            action = select_action(state, model, epsilon)
        new_state, reward, done, _ = env.step([actions[action]])
        modified_reward = reward + gamma * potential(new_state) - potential(state)
        memory.push((state, action, modified_reward, new_state, done))
        if done:
            state = env.reset()
        else:
            state = new_state

        if i > batch_size and i % 10 == 0:
            fit(memory.sample(batch_size))
        if i % update_period == 0:
            target_model = copy.deepcopy(model)
            print("{} / {}".format(i, max_iter))
            r, mr = render(render=i % render_period == 0)
            rewards.append(r)
            modified_rewards.append(mr)

    return rewards, modified_rewards


rewards, modified_rewards = teach(50000, 2000, 10000)

pylab.subplot(1, 1, 1)
xs = np.array(list(range(1, len(rewards) + 1))) * 2000
pylab.plot(xs, rewards, label="reward")
pylab.plot(xs, modified_rewards, label="modified reward")
pylab.grid()
pylab.legend()
# pylab.savefig('graph.png')
pylab.show()
