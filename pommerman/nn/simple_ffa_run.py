'''An example to show how to set up an pommerman game programmatically'''
import numpy as np

import sys
#sys.path.append('c:/Master_WINF/3_Semester/KI_Prak/Pommerman_wanja/pommerman_il/pommerman')


import pommerman 
from pommerman.agents import agent007, simple_agent
import random

from pommerman.nn import imitation_net
from pommerman.nn import a2c
from pommerman.nn import utils



def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''

    # Create a set of agents (exactly four)
    agent_list = [
        agent007.Agent007(),
        agent007.Agent007(),
        agent007.Agent007(),
        agent007.Agent007(),
    ]
    

    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    model = a2c.A2CNet()

    tranform_obj = utils.obsToPlanes(11)

    # Run the episodes just like OpenAI Gym
    num_episodes = 20
    wins = 0
    nn_inputs = []
    nn_targets = []
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        agent_pos = random.randint(0, 3)
        while not done:
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            if actions[agent_pos] != 0:
                nn_input = imitation_net.get_nn_input(state[agent_pos], tranform_obj)
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