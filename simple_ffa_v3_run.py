'''An example to show how to set up an pommerman game programmatically'''
import numpy as np

import sys
# sys.path.append('c:/Master_WINF/3_Semester/KI_Prak/Pommerman_wanja/pommerman_il/pommerman')


import pommerman
from pommerman.agents import *
import random

from pommerman.nn import imitation_net
from pommerman.nn import a2c_rl
from pommerman.nn import utils
from pommerman import constants


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''

    # Create a set of agents (exactly four)
    agent_list = [
        # agent007.Agent007(),
        # simple_agent.SimpleAgent(),
        agent007.Agent007(),
        # simple_agent.SimpleAgent(),
        stoner_agent.StonerAgent(),
    ]

    env = pommerman.make('DodgeBoard-v0', agent_list)
    #env = pommerman.make('KillBoard-v0', agent_list)
    model = a2c.A2CNet()

    tranform_obj = utils.obsToPlanes(constants.DODGE_BOARD_SIZE) # TODO variable

    # Run the episodes just like OpenAI Gym
    num_episodes = 50
    wins = 0
    nn_inputs = []
    nn_targets = []
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        agent_pos = 0
        last_bomb_spawned = 0  # damit nicht jede runde eine Bombe gespwaned wird
        spawn_bomb_every_x = 5

        while not done:
            env.render()
            pre_actions = env.act(state)
            actions = []
            for action in pre_actions:
                if isinstance(action, list):
                    actions.append(action[0])
                else:
                    actions.append(action)
            state, reward, done, info = env.step(actions)
            if actions[agent_pos] != 0:
                post_actions = []
                for action in actions:
                    post_actions.append([action, 0, 0])
                nn_input = imitation_net.get_nn_input(state[agent_pos], tranform_obj)
                nn_inputs += [nn_input]
                nn_target = imitation_net.get_nn_target(post_actions, agent_pos)
                nn_targets += [nn_target]

            last_bomb_spawned += 1
            if last_bomb_spawned % spawn_bomb_every_x == 0:
                env.make_bomb_board()
        print('Episode {} finished'.format(i_episode + 1))
        if info['result'].value == 0 and agent_pos in info['winners']:
            win = 1
        else:
            win = 0
        wins += win

        print("winrate: " + str(wins / (i_episode + 1)))
    env.close()

    print("final winrate: " + str(wins / num_episodes))

    imitation_net.train_net(model, nn_inputs, nn_targets)

    return wins / num_episodes


if __name__ == '__main__':
    main()
