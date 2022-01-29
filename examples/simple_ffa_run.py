'''An example to show how to set up an pommerman game programmatically'''
import numpy as np

import pommerman
from pommerman import agents
import random
from pommerman.nn import imitation_net
from pommerman.nn import a2c


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''

    # Create a set of agents (exactly four)
    agent_list = [
        agents.Agent007(),
        agents.SimpleAgent(),
        agents.Agent007(),
        agents.SimpleAgent(),
    ]
    agent_pos = 0

    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    model = a2c.A2CNet()

    # Run the episodes just like OpenAI Gym
    num_episodes = 1
    wins = 0
    nn_inputs = []
    nn_targets = []
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            nn_input = imitation_net.get_nn_input(state)
            nn_inputs += [nn_input]
            nn_target = imitation_net.get_nn_target(actions, agent_pos)
            nn_targets += [nn_target]
        print('Episode {} finished'.format(i_episode + 1))
        if info['result'].value == 0 and agent_pos in info['winners']:
            win = 1
        else:
            win = 0
        wins += win
        print("winrate: " + str(wins/(i_episode + 1)))
    env.close()

    print("final winrate: " + str(wins/num_episodes))

    imitation_net.train_net(model, nn_inputs, nn_targets)

    return wins/num_episodes


if __name__ == '__main__':
    main()
