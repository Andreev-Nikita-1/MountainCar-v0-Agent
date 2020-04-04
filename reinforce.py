import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np

from learning_visualization import learning_visualization

GAMES_BATCH_SIZE = 50
GAMMA = 0.998
LEARNING_RATE = 0.0001
GAME_NAME = 'MountainCar-v0'
INPUT_DIM = gym.make(GAME_NAME).observation_space.shape[0]
OUTPUT_DIM = gym.make(GAME_NAME).action_space.n
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def potential(state):
    return 3000 * np.abs(state[1]) + (100 if state[0] >= 0.5 else 0)


# будет возвращать вероятности действий для входного состояния
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, OUTPUT_DIM)
        )

        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

        self.apply(init_weights)
        self.train()
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        return F.softmax(self.model(x))

    # для батча х будет возвращать номер действия и логарифм вероятности
    def action(self, x):
        x = torch.tensor(x).to(device).float()
        ps = self(x)
        action = [np.random.choice(list(range(OUTPUT_DIM)), p=p) for p in ps.cpu().detach().numpy()]
        return action, torch.log(ps.gather(1, torch.tensor(action).unsqueeze(1).to(device).long()))

    def update(self, q_values, log_probs):
        log_probs = torch.stack(log_probs).view(-1, 1)
        q_values = torch.tensor(q_values).to(device).float()
        q_values = ((q_values - q_values.mean()) / q_values.std()).view(-1, 1)
        self.optimizer.zero_grad()
        loss = (-log_probs * q_values).sum()
        loss.backward()
        self.optimizer.step()


model = Model()


# будет хранить последовательность наград r_i для одной игры
class Protocol:
    def __init__(self):
        self.rewards = []
        self.q_values = []

    def add(self, r):
        self.rewards.append(r)

    # вычисляет Q(s_i, a_i) по формуле из уравнения Беллмана
    def accumulate(self):
        self.q_values = [self.rewards[-1]]
        for r in self.rewards[::-1][1:]:
            self.q_values.append(r + GAMMA * self.q_values[-1])
        self.q_values = self.q_values[::-1]
        return np.array(self.q_values)


# нужно для графиков
games = []
total_rewards = []


# параллельно играет GAMES_BATCH_SIZE игр,
# (применяет модель для вектора состояний,
# также постарался сделать чтобы градиенты тоже считались векторно, но точно не знаю тонкостей autograd
# я подумал что на gpu так будет лучше, хотя тут маленькие числа и у меня на cpu работает быстрее)
# делая по step_number шагов, затем обновляет модель с помощью полученных данных
# делает так epochs_number раз
def teach(steps_number, epochs_number):
    for epoch in range(epochs_number):
        envs = [gym.make(GAME_NAME) for _ in range(GAMES_BATCH_SIZE)]
        states = [env.reset() for env in envs]
        games_rewards = np.zeros(len(envs))
        protocols = [Protocol() for _ in envs]
        last_ends = -np.ones(len(envs), dtype=int)
        q_matrix = np.zeros((steps_number, len(envs)))
        all_actions = []

        total_reward = 0
        games_played = 0

        current_game = []

        for step in range(steps_number):
            actions, log_probs = model.action(states)
            all_actions.append(log_probs)
            new_states = []
            for i in range(len(envs)):
                new_state, reward, done, _ = envs[i].step(actions[i])

                # на графике рисуется первая игра из батча
                if i == 0:
                    current_game.append((states[0], actions[0]))

                modified_reward = reward + GAMMA * potential(new_state) - potential(states[i])
                games_rewards[i] += modified_reward
                protocols[i].add(modified_reward)

                if done:
                    if i == 0:
                        games.append((current_game, "epoch {}".format(epoch), epoch))
                        current_game = []
                    games_played += 1
                    total_reward += games_rewards[i]
                    games_rewards[i] = 0
                    q_matrix[last_ends[i] + 1:step + 1, i] = protocols[i].accumulate()
                    last_ends[i] = step
                    protocols[i] = Protocol()
                    new_state = envs[i].reset()
                new_states.append(new_state)

            states = new_states

        total_rewards.append(total_reward / games_played)
        model.update(q_matrix.flatten(), all_actions)
        print("epoch", epoch, ",  average reward =", total_reward / games_played)


teach(200, 1000)
learning_visualization(games, total_rewards)
