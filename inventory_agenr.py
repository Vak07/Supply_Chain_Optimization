import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class InventoryEnv:
    def __init__(self):
        self.state = [100, 50]  # [current inventory, demand forecast]
        self.action_space = [i for i in range(100)]  # Possible reorder quantities
        self.max_inventory = 200
        self.holding_cost = 1
        self.stockout_cost = 10

    def step(self, action):
        inventory, demand = self.state
        new_inventory = min(self.max_inventory, inventory + action)
        stockout_penalty = max(0, demand - new_inventory) * self.stockout_cost
        holding_cost = new_inventory * self.holding_cost
        reward = -(holding_cost + stockout_penalty)
        new_demand = np.random.randint(50, 100)  # New demand
        self.state = [new_inventory, new_demand]
        return self.state, reward

# Q-Learning Agent
class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
