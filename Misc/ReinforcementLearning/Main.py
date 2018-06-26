from BipedalWalker import *

if __name__ == '__main__':
    agent = Agent()
    # agent.load()
    for _ in range(10):
        agent.train(10)
        agent.save()
        agent.play(1)
