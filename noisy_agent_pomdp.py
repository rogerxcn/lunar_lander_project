import collections
import random
import json
import math

import itertools
import numpy as np
import matplotlib.pyplot as plt

import lunar_lander as lander


def state_extractor(s):
    state = (min(1, max(-1, int((s[0]) / 0.1))), \
            min(1, max(-1, int((s[1]) / 0.1))), \
            min(2, max(-2, int((s[2]) / 0.1))), \
            min(2, max(-2, int((s[3]) / 0.1))), \
            min(2, max(-2, int((s[4]) / 0.1))), \
            min(2, max(-2, int((s[5]) / 0.1))), \
            int(s[6]), \
            int(s[7]))

    return state


def state_space(s):
    space = []

    for x in np.linspace(s[0]-0.3, s[0]+0.3, num=10):
        space.append((x, s[1], s[2], s[3], s[4], s[5], s[6], s[7]))

    return space


def get_belief_vec(s):
    def pdf(mean, std, value):
        u = float(value - mean) / abs(std)
        y = (1.0 / (math.sqrt(2 * math.pi) * abs(std))) * math.exp(-u * u / 2.0)
        return y

    ss = state_space(s)
    bv = np.zeros(len(ss))

    for i in range(len(ss)):
        bs = ss[i]
        px = pdf(bs[0], 0.1, s[0])

        bv[i] = px

    bv = bv / np.sum(bv)

    return bv

def get_alpha_vec(Q, s):
    ss = state_space(s)
    av = np.zeros((len(ss), 4))

    for i in range(len(ss)):
        for a in range(4):
            av[i][a] = Q[sa_key(state_extractor(ss[i]), a)]

    return av


def sa_key(s, a):
    return str(s) + " " + str(a)


def policy_explorer(s, Q):
    bv = get_belief_vec(s)
    # print(bv)
    av = get_alpha_vec(Q, s)

    a = np.argmax(bv.T.dot(av))

    # print(av, bv, a)

    return a


def noisy_lander(env, Q, seed=None, render=False, num_iter=50, seg=50):
    def get_obs(true_loc):
        tX = true_loc[0]
        tY = true_loc[1]

        nX = np.random.normal(loc=tX, scale=0.1)

        return (nX, tY)

    r_seq = []
    it_reward = []
    Q = collections.defaultdict(float, Q)

    for it in range(num_iter):
        # initialize variables
        total_reward = 0
        steps = 0

        # reset environment
        s = env.reset()
        obs = get_obs((s[0], s[1]))
        s = (obs[0], obs[1], s[2], s[3], s[4], s[5], s[6], s[7])

        a = policy_explorer(s, Q)

        while True:
            s, r, done, info = env.step(a)

            obs = get_obs((s[0], s[1]))
            s = (obs[0], obs[1], s[2], s[3], s[4], s[5], s[6], s[7])
            a = policy_explorer(s, Q)

            total_reward += r

            if render and it % seg == 0:
                still_open = env.render()
                if still_open == False: break

            # if steps % 20 == 0 or done:
            #     print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))

            steps += 1

            if done or steps > 1000:
                # if total_reward > 50:
                #     print(ds, a, total_reward)
                it_reward.append(total_reward)
                break

        if it % seg == 0:
            avg_rwd = np.mean(np.array(it_reward))
            print("#It: ", it, " avg reward: ", avg_rwd, " out of ", len(it_reward), " trials")
            it_reward = []
            r_seq.append(avg_rwd)
            print("Overall Performance: ", np.mean(np.array(r_seq)))

    return r_seq


def main():
    num_iter = 1000

    with open('results/sarsa_data/sarsa_Q_3XY.json') as json_file:
        Q = json.load(json_file)

    env = lander.LunarLander()
    r_seq = noisy_lander(env, Q, render=True, num_iter=num_iter, seg=10)

    y = np.array(r_seq)
    x = np.linspace(0, num_iter, y.shape[0])

    plt.plot(x, y, label='Noisy POMDP Agent reward')
    plt.savefig("results/noisy_pomdp_agent.png")

    np.savetxt("results/noisy_pomdp_agent.txt", y)




if __name__ == '__main__':
    main()
