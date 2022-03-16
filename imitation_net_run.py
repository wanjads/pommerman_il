'''
this module generates the data from the the Agent007 and starts the 
imitation net for position prediction
'''
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
        simple_agent.SimpleAgent(),
        agent007.Agent007(),
        simple_agent.SimpleAgent(),
    ]
    

    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    model = a2c.A2CNet()

    tranform_obj = utils.obsToPlanes(11)

    # Run the episodes just like OpenAI Gym
    num_episodes = 5
    wins = 0
    nn_inputs = []
    nn_targets = []
    nn_index = []
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        #agent_pos = random.randint(0, 3)
        if i_episode % 2 == 0:
            agent_pos = 0
        else:
            agent_pos = 2
        while not done:
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            if actions[agent_pos] != 0:
                nn_input = imitation_net.get_nn_input(state[agent_pos], tranform_obj)
                nn_inputs += [nn_input]
                nn_target = imitation_net.get_nn_target(actions, agent_pos)
                nn_targets += [nn_target]
                nn_index.append(1) if done else nn_index.append(0)
        print('Episode {} finished'.format(i_episode + 1))
        if info['result'].value == 0 and agent_pos in info['winners']:
            win = 1
        else:
            win = 0
        wins += win

        print("winrate: " + str(wins/(i_episode + 1)))
    env.close()
    #np.save("inputs.npy", np.array(nn_inputs))
    #np.save("targets.npy", np.array(nn_targets))

    print("final winrate: " + str(wins/num_episodes))

    #nn_inputs = np.load("inputs.npy")
    #nn_targets = np.load("targets.npy")


    imitation_net.train_net(model, nn_inputs, nn_targets)

    return wins/num_episodes


if __name__ == '__main__':
    main()
