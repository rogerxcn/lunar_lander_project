import collections
import random
import numpy as np
import matplotlib.pyplot as plt

import lunar_lander as lander


def incr_config():
    pos_incr = (-1.0, 1.0, 0.25)
    vel_incr = (-1.0, 1.0, 0.25)
    angle_incr = (-1.0, 1.0, 0.25)
    contact_incr = (0.0, 1.0, 1.0)

    incr = (pos_incr, pos_incr, vel_incr, vel_incr, vel_incr, angle_incr, contact_incr, contact_incr)

    num_states = []

    for i, v in enumerate(incr):
        num_states.append(int((v[1] - v[0]) / v[2]) + 1)

    return incr, num_states


def state_extractor(s, incr):
    state = []

    for i, v in enumerate(s):
        if v < incr[i][0]:
            v = incr[i][0]
        elif v > incr[i][1]:
            v = incr[i][1]
        state_id = int((v - incr[i][0]) / incr[i][2])
        state.append(state_id)

    return state


def sa_key(s, a):
    s.append(a)
    return str(s)


def policy_explorer(s, Q, iter):
    rand = np.random.randint(0, 101)

    threshold = 90

    if iter > 500:
        threshold = 15
    if iter > 2500:
        threshold = 5

    if rand > threshold:
        return np.argmax(np.array([Q[sa_key(s, 0)], Q[sa_key(s, 1)], Q[sa_key(s, 2)], Q[sa_key(s, 3)]]))
    else:
        return np.random.randint(0, 4)

def sarsa_lander(lr, env, seed=None, render=False, num_iter=50, seg=50):
    env.seed(238)

    incr, states_shape = incr_config()
    states_shape.append(4)

    Q = collections.defaultdict(float)
    discount = 0.9

    r_seq = []
    it_reward = []

    for it in range(num_iter):
        # initialize variables
        total_reward = 0
        steps = 0

        if num_iter % seg == 1:
            it_reward = []

        last_sa = None
        last_reward = None
        # reset environment
        s = env.reset()
        # discretize state
        ds = state_extractor(s, incr)
        # start Sarsa
        while True:
            # use a policy generator to guide sarsa exploration
            a = policy_explorer(ds, Q, it)
            # step and get feedback
            s, r, done, info = env.step(a)
            # update corresponding Q
            ds = state_extractor(s, incr)
            current_sa = sa_key(ds, a)
            if last_sa is not None:
                Q[last_sa] += lr*(last_reward + discount * Q[current_sa] - Q[last_sa])

            last_sa = current_sa
            last_reward = r

            total_reward += r

            if render and it % seg == 0:
                still_open = env.render()
                if still_open == False: break

            # if steps % 20 == 0 or done:
            #     print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))

            steps += 1

            if done or steps > 500:
                Q[last_sa] += lr*(last_reward - Q[last_sa])
                it_reward.append(total_reward)
                break

        if it % seg == 0:
            avg_rwd = np.mean(np.array(it_reward))
            print(len(it_reward))
            print("#It: ", it, " avg reward: ", avg_rwd)
            r_seq.append(avg_rwd)

            lr /= 2

    return Q, r_seq


def main():
    lr = 0.1
    num_iter = 10000

    env = lander.LunarLander()
    Q, r_seq = sarsa_lander(lr, env, render=True, num_iter=num_iter, seg=50)

    y = np.array(r_seq)
    x = np.linspace(0, num_iter, y.shape[0])

    plt.plot(x, y, label='Sarsa reward')
    plt.savefig("sarsa_reward.png")




if __name__ == '__main__':
    main()
