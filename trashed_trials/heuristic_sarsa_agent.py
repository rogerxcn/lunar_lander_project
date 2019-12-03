import collections
import random
import numpy as np
import matplotlib.pyplot as plt

import lunar_lander as lander

def heuristic(s):
    # Heuristic for:
    # 1. Testing.
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
    elif angle_todo < -0.05: a = 3
    elif angle_todo > +0.05: a = 1
    return a


def state_extractor(s):
    pos_base = 0.25
    vel_base = 0.25
    angle_base = 0.25

    state = (s[0] // pos_base, s[1] // pos_base, s[2] // vel_base, s[3] // vel_base, s[4] // vel_base, s[5] // angle_base, s[6], s[7])

    return state


def lr_scheduler(it):
    return 0.2



def sa_key(s, a):
    return str(s) + " " + str(a)


def policy_explorer(s, hs, Q, iter):
    rand = np.random.randint(0, 100)

    threshold = 100

    if iter > 500:
        threshold = 50
    if iter > 1000:
        threshold = 20
    if iter > 2000:
        threshold = 5
    if iter > 5000:
        threshold = 1

    if rand >= threshold:
        Qv = [ Q[sa_key(s, action)] for action in [0, 1, 2, 3]]
        return np.argmax(np.array(Qv))
    else:
        return heuristic(hs)




def sarsa_lander(env, seed=None, render=False, num_iter=50, seg=50):
    env.seed(42)

    Q = collections.defaultdict(float)
    gamma = 0.99

    r_seq = []
    it_reward = []

    for it in range(num_iter):
        # initialize variables
        total_reward = 0
        steps = 0

        lr = lr_scheduler(it)

        # reset environment
        s = env.reset()

        ds = state_extractor(s)
        a = policy_explorer(ds, s, Q, it)
        # start Sarsa
        while True:
            # use a policy generator to guide sarsa exploration
            # step and get feedback
            sp, r, done, info = env.step(a)
            # update corresponding Q
            dsp = state_extractor(sp)
            ap = policy_explorer(dsp, sp, Q, it)

            sa = sa_key(ds, a)
            next_sa = sa_key(dsp, ap)

            if not done:
                Q[sa] += lr*(r + gamma * Q[next_sa] - Q[sa])
            else:
                Q[sa] += lr*(r - Q[sa])

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
                it_reward.append(total_reward)
                break

        if it % seg == 0:
            avg_rwd = np.mean(np.array(it_reward))
            it_reward = []
            print("#It: ", it, " avg reward: ", avg_rwd)
            r_seq.append(avg_rwd)

    return Q, r_seq


def main():
    num_iter = 100000

    env = lander.LunarLander()
    Q, r_seq = sarsa_lander(env, render=True, num_iter=num_iter, seg=50)

    y = np.array(r_seq)
    x = np.linspace(0, num_iter, y.shape[0])

    plt.plot(x, y, label='Sarsa reward')
    plt.savefig("sarsa_reward.png")




if __name__ == '__main__':
    main()
