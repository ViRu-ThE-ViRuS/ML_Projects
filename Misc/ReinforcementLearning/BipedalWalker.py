import random
import _pickle as pickle
import numpy as np
from evostra import EvolutionStrategy
import gym


# uses custom optimized version of evostra library
# original link: https://github.com/alirezamika/evostra


class Model(object):
    def __init__(self):
        self.weights = [np.zeros(shape=(24, 16)), np.zeros(
            shape=(16, 16)), np.zeros(shape=(16, 4))]

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        out = out / np.linalg.norm(out)
        for layer in self.weights:
            out = np.dot(out, layer)
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:
    AGENT_HISTORY_LENGTH = 1
    POPULATION_SIZE = 20
    EPS_AVG = 1
    SIGMA = 0.1
    LEARNING_RATE = 0.01
    INITIAL_EXPLORATION = 1.0
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 1000000

    def __init__(self):
        self.env = gym.make('BipedalWalker-v2')
        self.model = Model()
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)
        self.exploration = self.INITIAL_EXPLORATION

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))
        return prediction

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)

    def play(self, episodes, render=True):
        self.model.set_weights(self.es.weights)
        for episode in range(episodes):
            total_reward = 0
            observation = self.env.reset()
            sequence = [observation] * self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                if render:
                    self.env.render()
                action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)
            print("total reward:", total_reward)

    def train(self, iterations):
        self.es.run(iterations, print_step=1)

    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)

        for episode in range(self.EPS_AVG):
            observation = self.env.reset()
            sequence = [observation] * self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration -
                                       self.INITIAL_EXPLORATION / self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)

        return total_reward / self.EPS_AVG


if __name__ == '__main__':
    agent = Agent()
    agent.load()
    # agent.train(100)  # train for 100 iterations
    # agent.save()

    # play one episode
    agent.play(1)
