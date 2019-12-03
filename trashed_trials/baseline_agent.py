import numpy as np

import lunar_lander as lander


def baseline_heuristic(env, s):
    # Useful lookup table for structure state:
    # state = [
    #     (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
    #     (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
    #     vel.x*(VIEWPORT_W/SCALE/2)/FPS,
    #     vel.y*(VIEWPORT_H/SCALE/2)/FPS,
    #     self.lander.angle,
    #     20.0*self.lander.angularVelocity/FPS,
    #     1.0 if self.legs[0].ground_contact else 0.0,
    #     1.0 if self.legs[1].ground_contact else 0.0
    #     ]
    action = np.random.randint(1, 3)


    if s[3] < -1:
            action = 2
    elif abs(s[0]) > 0.1:
        if s[0] > 0:
            action = 1
        if s[0] < 0:
            action = 3


    return action


def baseline_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = baseline_heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
    return total_reward


def main():
    baseline_lander(lander.LunarLander(), render=True)


if __name__ == '__main__':
    main()
