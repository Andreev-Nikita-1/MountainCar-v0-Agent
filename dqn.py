import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import copy
import random
import numpy as np

from learning_visualization import learning_visualization

GAMES_BATCH_SIZE = 50
GAMMA = 0.998
EPS_COEFF = 0.9995
LEARNING_RATE = 0.0001
GAME_NAME = 'MountainCar-v0'
INPUT_DIM = gym.make(GAME_NAME).observation_space.shape[0]
OUTPUT_DIM = gym.make(GAME_NAME).action_space.n
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

target_model = None


def potential(state):
    return 3000 * np.abs(state[1]) + (100 if state[0] >= 0.5 else 0)


# будет возвращать Q(a, s)
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

    # для батча x будет возвращать номер действия и значение Q
    def action(self, x, epsilon=0):
        x = torch.tensor(x).to(device).float()
        res = self(x)
        random_inds = [random.random() < epsilon for _ in range(len(x))]
        action = [torch.argmax(res[i]).item() if not random_inds[i] else np.random.randint(OUTPUT_DIM) for i in
                  range(len(x))]
        return action, res.gather(1, torch.tensor(action, device=device).unsqueeze(1))

    def forward(self, x):
        return self.model(x)

    def update(self, q_values, rewards, new_states):
        q_values = torch.stack(q_values)
        new_states = torch.tensor(new_states).to(device).float()
        rewards = torch.tensor(rewards).to(device).float()
        with torch.no_grad():
            target_q = target_model.action(new_states)[1].view(-1)
        target_q = rewards + target_q * GAMMA
        loss = F.smooth_l1_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


model = Model()
target_model = copy.deepcopy(model)

games = []
total_rewards = []


# параллельно играет GAMES_BATCH_SIZE игр,
# (применяет модель для вектора состояний,
# также постарался сделать чтобы градиенты тоже считались векторно, но точно не знаю тонкостей autograd
# я подумал что на gpu так будет лучше, хотя тут маленькие числа и у меня на cpu работает быстрее)
# каждые batch_size // GAMES_BATCH_SIZE шагов обновляет модель с помощью полученных данных
# всего делает steps_number шагов, затем обновляет target_model
# делает так epochs_number раз
def teach(steps_number, epochs_number, batch_size=GAMES_BATCH_SIZE):
    global target_model, model
    for epoch in range(epochs_number):
        print(epoch)
        envs = [gym.make(GAME_NAME) for _ in range(GAMES_BATCH_SIZE)]
        states = [env.reset() for env in envs]
        games_rewards = np.zeros(len(envs))
        total_reward = 0
        games_played = 0

        all_q = []
        all_r = []
        all_n = []

        current_game = []

        for step in range(steps_number):
            epsilon = EPS_COEFF ** (steps_number * epoch + step)
            actions, q_values = model.action(states, epsilon=epsilon)
            rewards = []
            next_states = []
            new_states = []
            for i in range(len(envs)):
                new_state, reward, done, _ = envs[i].step(actions[i])
                modified_reward = reward + GAMMA * potential(new_state) - potential(states[i])
                games_rewards[i] += modified_reward
                rewards.append(modified_reward)
                next_states.append(new_state)

                # на графике рисуется первая игра из батча
                if i == 0:
                    current_game.append((states[0], actions[0]))

                if done:
                    if i == 0:
                        games.append((current_game, "epoch {}".format(epoch), epoch))
                        current_game = []
                    games_played += 1
                    total_reward += games_rewards[i]
                    games_rewards[i] = 0
                    new_state = envs[i].reset()
                new_states.append(new_state)

            states = new_states
            all_q.append(q_values)
            all_r += rewards
            all_n += next_states
            if (steps_number * epoch + step) % (batch_size // GAMES_BATCH_SIZE) == 0:
                model.update(all_q, all_r, all_n)
                all_q, all_n, all_r = [], [], []

        total_rewards.append(total_reward / games_played)
        target_model = copy.deepcopy(model)
        print("epoch", epoch, ",  average reward =", total_reward / games_played)


teach(200, 500)
learning_visualization(games, total_rewards)
