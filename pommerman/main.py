from model import Leif
import colorama
from pommerman import agents
from collections import Counter
import time
import math
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt


# tobis anpassungen erfordern
from nn.a2c_v3 import A2CNet
import pommerman
import numpy as np
import torch
import torch.nn.functional as F

ROLLOUTS_PER_BATCH = 1
batch = []


class World:
    def __init__(self, init_gmodel=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if init_gmodel:
            self.gmodel = A2CNet()  # Global model
            self.gmodel = self.gmodel.to(self.device)
        self.gmodel = self.gmodel.to(self.device)
        self.model = self.gmodel.to(self.device)

        self.model = A2CNet()  # Agent (local) model
        print("device: ", self.device, " cuda av: ", torch.cuda.is_available(), " cuda device: ", torch.cuda.device(0), torch.cuda.device_count(), torch.cuda.get_device_name(0))
        self.model = self.model.to(self.device)

        self.leif = Leif(self.model)

        self.agent_list = [
            self.leif,
            agents.StonerAgent()
            #agents.SimpleAgent(),
            #agents.SimpleAgent(),
            #agents.SimpleAgent()
        ]
        self.env = pommerman.make('DodgeBoard-v0', self.agent_list)
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
    states, actions, probs, values = leif.clear()
    old_state = None
    last_action = 0

    spawn_bomb_every_x = random.randint(5,12)
    counter = 0

    while not done and 10 in state[0]['alive']:

        if do_print:
            time.sleep(0.1)
            os.system('clear')
            print(state[0]['board'])

        action = env.act(state)
        state, start_rewards, done, info = env.step(action)
        state[0]["can_kick"] = 1
        action = action[0]
        if old_state is None:
            old_state = state
        reward = get_reward(state, old_state, 0, action, last_action, done, counter)
        old_state = state
        last_action = action
        rewards.append(reward)
        dones.append(done)

        if counter % spawn_bomb_every_x == 0 and env.spec.id == "DodgeBoard-v0" and not done:
            print(counter)
            env.make_bomb_board()
        counter += 1

        if counter == 100: done = True


    print("rounds: ", counter)
    # dont need them without lstm so set them to empty
    hns, cns = [], []
    rewards = rewards[:len(values)]

    return (states.copy(),
            actions.copy(),
            rewards, dones,
            (hns, cns),
            probs.copy(),
            values.copy())


def get_reward(state, old_state, agent_nr, action, last_action, done, counter):
    # developer note: on the board:
    # 0: nothing, 1: unbreakable wall, 2: wall, 3: bomb, 4: flames, 6,7,8: pick-ups:  11,12 and 13: enemies
    reward = 0
    # penalty for dying
    #if 10 not in state[0]['alive'] and counter < 11:
    #                 reward -= 1
    #elif 10 not in state[0]['alive'] and counter < 15:
    #    reward -= 0.75
    if 10 not in state[0]['alive'] and done:
        reward -= 1
    elif done:
        reward += 1

    if counter == 10:
        reward += 0.1
    elif counter == 15:
        reward += 0.15
    elif counter == 25:
        reward += 0.25
    elif counter == 20:
        reward += 0.2
    elif counter == 30:
        reward += 0.3
    elif counter == 40:
        reward += 0.4
    elif counter == 50: reward += 0.5
    elif counter == 60: reward += 0.6
    elif counter == 75: reward += 0.75

    # actionfilter invalid actions
    if old_state[agent_nr]['position'] == state[agent_nr]['position'] and action != 0 and action != 5:
        reward -= 0.1
    if old_state[agent_nr]['position'] == state[agent_nr]['position'] and action != 0 and action == 5:
        reward -= 0.2
    if old_state[agent_nr]['position'] == state[agent_nr]['position'] and action == 0 and action != 5:
        if last_action == 0:
            reward -= 0.3

    if action in [1,2,3,4]:
        reward += 0.01
#
    ## reward agent for picking up power-ups
    blast_strength = state[agent_nr]['blast_strength']
    old_blast_strength = old_state[agent_nr]['blast_strength']
    can_kick = int(state[agent_nr]['can_kick'])
    old_can_kick = int(old_state[agent_nr]['can_kick'])
    reward += (can_kick-old_can_kick)*0.2
    reward += (blast_strength-old_blast_strength)*0.2
    return reward


def gmodel_train(gmodel, states, hns, cns, actions, rewards, gae):
    states = torch.stack(states)
    gmodel.train()
    values, probs, _, _ = gmodel(states.to(gmodel.device))

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
        states.extend(torch.tensor(np.array(s)))
        actions.extend(a)
        rewards.extend(gmodel.discount_rewards(r))

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
    path = "./saved_models"
    if not os.path.exists(path):
        os.makedirs(path)
    model, gmodel = world.model, world.gmodel
    leif, env = world.leif, world.env

    if os.path.isfile("saved_models/torch_state.tar"):  # turn off for new model
        model.load_state_dict(torch.load("saved_models/torch_state.tar")["model_state_dict"])
        gmodel.load_state_dict(torch.load("saved_models/torch_state.tar")["model_state_dict"])
        print("loaded checkpoint")
    print(os.path.isfile("saved_models/torch_state.tar"))

    if os.path.exists("training.txt"):
        os.remove("training.txt")

    reward_list = []

    rr = 0
    ii = 0
    for i in range(1000000):
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
        reward_list.append(np.mean(rewards))
        if i >= 1000 and i % 1000 == 0:
            path_2 = os.path.join(path, "torch_state.tar")
            torch.save({'model_state_dict': model.state_dict(),
                            # TODO: not sure if self.optimizer.state_dict is the same thing
                            # that timour also saves (Copied from timour)
                            'optimizer_state_dict': gmodel.optimizer.state_dict(),
                            }, path_2)
            dummy_input = torch.ones(1, 18, 11, 11, dtype=torch.float)
            dummy_input = dummy_input.to(world.gmodel.device)
            input_names = ["data"]
            output_names = ["value_out", "policy_out"]
            torch.onnx.export(model, dummy_input, str(path / Path(f"model-bsize-{1}.onnx")), input_names=input_names,
                      output_names=output_names)
            print("saved weights")
    print(reward_list, list(range(i+1)))
    plot(list(range(i+1)),reward_list)





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
        model.load_state_dict(torch.load("saved_models/torch_state.tar")["model_state_dict"])
        done, state, _ = False, env.reset(), leif.clear()
        t = 0
        while not done:
            if do_print:
                time.sleep(0.1)
                print(state[0]['board'])
                print("\n\n")
                print("Probs: \t", leif.probs[-1] if len(leif.probs) > 0 else [])
                print("Val: \t", leif.values[-1] if len(leif.values) > 0 else None)
                print("\nReward: ", reward, "Time", t)

            action = env.act(state)
            state, reward, done, info = env.step(action)
            t += 1

def plot(x, y):
    plt.xlabel("epochs")
    plt.ylabel("avg rewards")
    plt.plot(x,y)
    plt.show(block=False)


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(42)
    random.seed(42)
    #evaluate(World())
    train(World())

if __name__ == "__main__":
    main()
