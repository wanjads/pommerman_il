from model import Leif
#from model import A2CNet
import colorama
from pommerman import agents
from collections import Counter
import time
import math
import os
import numpy.matlib


# tobis anpassungen erfordern
from nn.a2c_rl import A2CNet
import pommerman
import numpy as np
import torch
import torch.nn.functional as F

ROLLOUTS_PER_BATCH = 1
batch = []


class World:
    def __init__(self, init_gmodel=True):
        if init_gmodel:
            self.gmodel = A2CNet()  # Global model

        self.model = A2CNet()  # Agent (local) model # TODO change to our A2cNet
        self.leif = Leif(self.model)

        self.agent_list = [
            self.leif,
            agents.StonerAgent()
            #agents.SimpleAgent(),
            #agents.SimpleAgent(),
            #agents.SimpleAgent()
        ]
        self.env = pommerman.make('BombBoard-v0', self.agent_list)
        fmt = {
            'int': self.color_sign,
            'float': self.color_sign
        }
        np.set_printoptions(formatter=fmt, linewidth=300)
        pass

    def color_sign(self, x):
        if x == 0:
            c = colorama.Fore.LIGHTBLACK_EX
        elif x == 1:
            c = colorama.Fore.BLACK
        elif x == 2:
            c = colorama.Fore.BLUE
        elif x == 3:
            c = colorama.Fore.RED
        elif x == 4:
            c = colorama.Fore.RED
        elif x == 10:
            c = colorama.Fore.YELLOW
        else:
            c = colorama.Fore.WHITE
        x = '{0: <2}'.format(x)
        return f'{c}{x}{colorama.Fore.RESET}'


def do_rollout(env, leif, do_print=False):
    done, state = False, env.reset()
    rewards, dones = [], []
    states, actions, hidden, probs, values = leif.clear()
    old_state = None
    last_action = 0

    last_bomb_spawned = 0  # damit nicht jede runde eine Bombe gespwaned wird
    spawn_bomb_every_x = 5

    while not done and 10 in state[0]['alive']:
        env.render()
        if do_print:
            time.sleep(0.1)
            os.system('clear')
            print(state[0]['board'])

        action = env.act(state)
        state, start_rewards, done, info = env.step(action)
        action = action[0]
        if old_state is None:
            old_state = state
        reward = get_reward(state, old_state, 0, action, last_action)
        # print(str(state[0]['position']) + str(old_state[0]['position']) + str(reward))
        old_state = state
        last_action = action
        rewards.append(reward)
        dones.append(done)


        last_bomb_spawned += 1
        if last_bomb_spawned % spawn_bomb_every_x == 0 and env.spec.id == "DodgeBoard-v0":
            env.make_bomb_board()

    hidden = hidden[:-1].copy()
    hns, cns = [], []
    for hns_cns_tuple in hidden:
        hns.append(hns_cns_tuple[0])
        cns.append(hns_cns_tuple[1])

    rewards = rewards[:len(values)]

    return (states.copy(),
            actions.copy(),
            rewards, dones,
            (hns, cns),
            probs.copy(),
            values.copy())


def get_reward(state, old_state, agent_nr, action, last_action):
    # developer note: on the board:
    # 0: nothing, 1: unbreakable wall, 2: wall, 3: bomb, 4: flames, 6,7,8: pick-ups:  11,12 and 13: enemies
    reward = 0
    # penalty for dying
    if 10 not in state[0]['alive']:
        reward -= 1

    # reward stage 0:
    # teach the agent to move and not make invalid actions (move into walls, place bombs when you have no ammo)
    ammo = old_state[agent_nr]['ammo']
    if action != 5:
        if state[agent_nr]['position'] == old_state[agent_nr]['position']:
            reward -= 0.03
    elif ammo == 0:
        reward -= 0.03

    # reward stage 1: teach agent to bomb walls (and enemies)
    # compute adjacent squares
    position = state[agent_nr]['position']
    adj = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not ((i == j) or i + j == 0)]
    adjacent = numpy.matlib.repmat(position, 4, 1)
    adjacent = adjacent - np.asarray(adj)
    # limit adjacent squares to only include inside board
    adjacent = np.clip(adjacent, 0, 10)
    if action == 5 and ammo > 0:
        board = state[agent_nr]['board']
        for xy in adjacent:
            square_val = board[xy[0]][xy[1]]
            if square_val == 2:
                reward += 0.2
            elif square_val == 11 or square_val == 12 or square_val == 13:
                reward += 0.5

    # reward stage2: teach agent to not stand on or beside bombs
    # reward /= 4
    bomb_life = state[agent_nr]['bomb_life']
    # if we stand on a bomb or next to bomb
    just_placed_bomb = np.logical_xor(last_action == 5, action == 5)
    if bomb_life[position] > 0 and not just_placed_bomb:
        reward -= 0.1 * (9-bomb_life[position])
    for xy in adjacent:
        if bomb_life[xy[0]][xy[1]] > 0:
            reward -= 0.05 * (9-bomb_life[xy[0]][xy[1]])

    # reward agent for picking up power-ups
    blast_strength = state[agent_nr]['blast_strength']
    old_blast_strength = old_state[agent_nr]['blast_strength']
    can_kick = int(state[agent_nr]['can_kick'])
    old_can_kick = int(old_state[agent_nr]['can_kick'])
    reward += (can_kick-old_can_kick)*0.02
    # reward += (max_ammo-old_max_ammo)*0.02 #TODO, see arguments
    reward += (blast_strength-old_blast_strength)*0.02
    return reward


def gmodel_train(gmodel, states, hns, cns, actions, rewards, gae):
    states, hns, cns = torch.stack(states), torch.stack(hns, dim=0), torch.stack(cns, dim=0)
    gmodel.train()
    probs, values, _, _ = gmodel(states.to(gmodel.device), hns.to(gmodel.device), cns.to(gmodel.device))

    prob = F.softmax(probs, dim=-1)
    log_prob = F.log_softmax(probs, dim=-1)
    entropy = -(log_prob * prob).sum(1)

    log_probs = log_prob[range(0, len(actions)), actions]
    advantages = torch.tensor(rewards).to(gmodel.device) - values.squeeze(1)
    value_loss = advantages.pow(2) * 0.5
    policy_loss = -log_probs * torch.tensor(gae).to(gmodel.device) - gmodel.entropy_coef * entropy

    gmodel.optimizer.zero_grad()
    pl = policy_loss.sum()
    vl = value_loss.sum()
    loss = pl + vl
    loss.backward()
    gmodel.optimizer.step()

    return loss.item(), pl.item(), vl.item()


def unroll_rollouts(gmodel, list_of_full_rollouts):
    gamma = gmodel.gamma
    tau = 1

    states, actions, rewards, hns, cns, gae = [], [], [], [], [], []
    for (s, a, r, d, h, p, v) in list_of_full_rollouts:
        states.extend(torch.tensor(s))
        actions.extend(a)
        rewards.extend(gmodel.discount_rewards(r))

        hns.extend([torch.tensor(hh) for hh in h[0]])
        cns.extend([torch.tensor(hh) for hh in h[1]])

        # Calculate GAE
        last_i, _gae, __gae = len(r) - 1, [], 0
        for i in reversed(range(len(r))):
            next_val = v[i + 1] if i != last_i else 0
            delta_t = r[i] + gamma * next_val - v[i]
            __gae = __gae * gamma * tau + delta_t
            _gae.insert(0, __gae)

        gae.extend(_gae)

    return states, hns, cns, actions, rewards, gae


def train(world):
    model, gmodel = world.model, world.gmodel
    leif, env = world.leif, world.env

    if os.path.isfile("convrnn-s.weights"):  # turn off for new model
        model.load_state_dict(torch.load("convrnn-s.weights", map_location='cpu'))
        gmodel.load_state_dict(torch.load("convrnn-s.weights", map_location='cpu'))
        print("loaded checkpoint")

    if os.path.exists("training.txt"):
        os.remove("training.txt")

    rr = 0
    ii = 0
    for i in range(3001):
        full_rollouts = [do_rollout(env, leif) for _ in range(ROLLOUTS_PER_BATCH)]
        states, hns, cns, actions, rewards, gae = unroll_rollouts(gmodel, full_rollouts)
        gmodel.gamma = 0.5 + 1 / 2. / (1 + math.exp(-0.0003 * (i - 20000)))  # adaptive gamma
        l, pl, vl = gmodel_train(gmodel, states, hns, cns, actions, rewards, gae)
        rr = rr * 0.99 + (np.mean(rewards) / len(actions)) / ROLLOUTS_PER_BATCH * 0.01
        ii += len(actions)
        print(i, "\t", round(gmodel.gamma, 3), round(rr*1000, 3), "\twins:", "---", Counter(actions),
              round(sum(rewards), 3), round(l, 3), round(pl, 3), round(vl, 3))
        with open("training.txt", "a") as f:
            print(rr, "\t", round(gmodel.gamma, 4), "\t", round(vl, 3), "\t", round(pl, 3), "\t", round(l, 3), file=f)
        model.load_state_dict(gmodel.state_dict())
        if i >= 10 and i % 30 == 0:
            torch.save(gmodel.state_dict(), "convrnn-s.weights")
            print("saved weights")


def run(world):
    done, ded, state, _ = False, False, world.env.reset(), world.leif.clear()

    while not done:
        action = world.env.act(state)
        state, reward, done, info = world.env.step(action)
        print(world.leif.board_cent)
        print(world.leif.bbs_cent)
        print(world.leif.bl_cent)
        time.sleep(0.2)

    world.env.close()
    return None


def evaluate(world):
    env = world.env
    model = world.model
    leif = world.leif
    leif.debug = True
    leif.stochastic = True

    do_print = True
    reward = 0

    while True:
        model.load_state_dict(torch.load("convrnn-s.weights", map_location='cpu')) # datei nicht vorhanden

        done, state, _ = False, env.reset(), leif.clear()
        t = 0
        while not done:
            env.render()
            if do_print:
                time.sleep(0.1)
                # os.system('clear')
                print(state[0]['board'])
                print("\n\n")
                print("Probs: \t", leif.probs[-1] if len(leif.probs) > 0 else [])
                print("Val: \t", leif.values[-1] if len(leif.values) > 0 else None)
                print("\nReward: ", reward, "Time", t)

            action = env.act(state)
            state, reward, done, info = env.step(action)
            t += 1


#evaluate(World())
train(World())
