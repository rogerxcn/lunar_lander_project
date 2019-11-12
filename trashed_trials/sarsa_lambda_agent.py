import collections
import random
import numpy as np
import matplotlib.pyplot as plt

import lunar_lander as lander


def state_extractor(s):

    # state = (int((s[0] - 0.3 / 2.0) / 0.3), \
    #         int((s[1] - 0.3 / 2.0) / 0.3), \
    #         int((s[2] - 0.4 / 2.0) / 0.4), \
    #         int((s[3] - 0.4 / 2.0) / 0.4), \
    #         int((s[4] - 0.2 / 2.0) / 0.2), \
    #         int((s[5] - 0.4 / 2.0) / 0.4), \
    #         int(s[6]), \
    #         int(s[7]))

    state = (int((s[0]) / 0.3), \
            int((s[1]) / 0.3), \
            int((s[2]) / 0.4), \
            int((s[3]) / 0.4), \
            int((s[4]) / 0.1), \
            int((s[5]) / 0.1), \
            int(s[6]), \
            int(s[7]))

    # print(s[4], s[5])

    return state


def lr_scheduler(it):
    return 0.3


def sa_key(s, a):
    return str(s) + " " + str(a)


def policy_explorer(s, Q, iter):
    rand = np.random.randint(0, 100)

    threshold = 50
    if iter > 100:
        threshold = 20
    if iter > 5000:
        threshold = 10

    if rand >= threshold:
        Qv = np.array([ Q[sa_key(s, action)] for action in [0, 1, 2, 3]])
        return np.argmax(Qv)
    else:
        return np.random.randint(0, 4)




def sarsa_lander(env, seed=None, render=False, num_iter=50, seg=50):
    env.seed(42)

    Q = collections.defaultdict(float)

    gamma = 0.95
    decay = 0.2

    r_seq = []
    it_reward = []

    for it in range(num_iter):
        # initialize variables
        total_reward = 0
        steps = 0

        lr = lr_scheduler(it)

        # reset environment
        s = env.reset()

        credits = collections.defaultdict(float)

        ds = state_extractor(s)
        a = policy_explorer(ds, Q, it)
        # start Sarsa
        while True:
            # use a policy generator to guide sarsa exploration
            # step and get feedback
            sa = sa_key(ds, a)

            credits[sa] += 1.0

            sp, r, done, info = env.step(a)
            # update corresponding Q
            dsp = state_extractor(sp)
            ap = policy_explorer(dsp, Q, it)

            next_sa = sa_key(dsp, ap)

            temporal_diff = r + gamma * Q[next_sa] - Q[sa]

            if done:
                Q[next_sa] += lr*(r - Q[next_sa])

            for key in credits:
                Q[key] = lr * temporal_diff * credits[key]
                credits[key] *= decay * gamma


            ds = dsp
            a = ap

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

    return Q, r_seq


def main():
    num_iter = 10000

    env = lander.LunarLander()
    Q, r_seq = sarsa_lander(env, render=True, num_iter=num_iter, seg=100)

    y = np.array(r_seq)
    x = np.linspace(0, num_iter, y.shape[0])

    plt.plot(x, y, label='Sarsa Lambda reward')
    plt.savefig("sarsa_lambda_reward.png")

    np.savetxt("sarsa_lambda_reward.txt", y)




if __name__ == '__main__':
    main()
