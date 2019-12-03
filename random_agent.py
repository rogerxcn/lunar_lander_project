import collections
import random
import numpy as np
import matplotlib.pyplot as plt

import lunar_lander as lander


def random_lander(env, seed=None, render=False, num_iter=50, seg=50):
    env.seed(42)

    r_seq = []
    it_reward = []

    for it in range(num_iter):
        # initialize variables
        total_reward = 0
        steps = 0

        # reset environment
        s = env.reset()
        # start Sarsa
        while True:
            # use a policy generator to guide sarsa exploration
            # step and get feedback
            a = np.random.randint(0, 4)

            sp, r, done, info = env.step(a)

            total_reward += r

            if render:
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

    return r_seq


def main():
    num_iter = 10000

    env = lander.LunarLander()
    r_seq = random_lander(env, render=False, num_iter=num_iter, seg=100)

    y = np.array(r_seq)
    x = np.linspace(0, num_iter, y.shape[0])

    plt.plot(x, y, label='Random reward')
    plt.savefig("results/random_reward.png")

    np.savetxt("results/random_reward.txt", y)




if __name__ == '__main__':
    main()
