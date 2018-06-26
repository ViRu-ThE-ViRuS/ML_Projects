from BipedalWalker import *

if __name__ == '__main__':
    agent = Agent()
    # agent.load('weights.pkl')
    agent.train(100)
    agent.save()
    agent.play(1)
