import collections
import random
import json

import numpy as np
import matplotlib.pyplot as plt

import lunar_lander as lander


def state_extractor(s):
    state = (min(3, max(-3, int((s[0]) / 0.05))), \
            min(3, max(-1, int((s[1]) / 0.1))), \
            min(2, max(-2, int((s[2]) / 0.1))), \
            min(2, max(-2, int((s[3]) / 0.1))), \
            min(2, max(-2, int((s[4]) / 0.1))), \
            min(2, max(-2, int((s[5]) / 0.1))), \
            int(s[6]), \
            int(s[7]))

    return state


def sa_key(s, a):
    return str(s) + " " + str(a)


def policy_explorer(s, Q):
    Qv = np.array([ Q[sa_key(s, action)] for action in [0, 1, 2, 3]])
    return np.argmax(Qv)


def noisy_lander(env, Q, seed=None, render=False, num_iter=50, seg=50):
    r_seq = []
    it_reward = []
    Q = collections.defaultdict(float, Q)

    for it in range(num_iter):
        # initialize variables
        total_reward = 0
        steps = 0

        # reset environment
        s = env.reset()

        ds = state_extractor(s)
        a = policy_explorer(ds, Q)
        # start Sarsa
        while True:
            # use a policy generator to guide sarsa exploration
            # step and get feedback
            sa = sa_key(ds, a)

            sp, r, done, info = env.step(a)
            # update corresponding Q
            ds = state_extractor(sp)
            a = policy_explorer(ds, Q)

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
    num_iter = 200

    with open('results/sarsa_data/sarsa_Q_7X5Y.json') as json_file:
        Q = json.load(json_file)

    env = lander.LunarLander()
    r_seq = noisy_lander(env, Q, render=False, num_iter=num_iter, seg=10)

    y = np.array(r_seq)
    x = np.linspace(0, num_iter, y.shape[0])

    plt.plot(x, y, label='7X5Y Agent reward')
    plt.savefig("results/7X5Y_agent.png")

    np.savetxt("results/7X5Y_agent.txt", y)




if __name__ == '__main__':
    main()
