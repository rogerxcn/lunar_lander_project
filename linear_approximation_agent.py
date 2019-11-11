import numpy as np

import lunar_lander as lander


def linear_approximation_lander(theta, lr, env, seed=None, render=False, num_iter=50):
    env.seed(seed)
    total_reward = 0
    steps = 0

    for it in range(num_iter):
        s = env.reset()
        s = np.append(s, 1.0)

        while True:
            predicted_reward = theta.dot(s.T)

            eps = np.random.randint(0, 100)
            if eps > 20:
                a = np.argmax(predicted_reward)
            else:
                a = np.random.randint(0, 4)

            s, r, done, info = env.step(a)

            s = np.append(s, 1.0)
            r -= steps / 100

            theta[a] += lr * (r - predicted_reward[a]) * s

            total_reward += r

            if render:
                still_open = env.render()
                if still_open == False: break

            if steps % 20 == 0 or done:
                print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            if done: break
            
    return theta


def main():
    lr = 1e-2
    theta = np.random.randn(4, 9) / 100.0

    env = lander.LunarLander()
    theta = linear_approximation_lander(theta, lr, env, render=True, num_iter=50)

    print("Final theta: ", theta)
    np.savetxt("weights/linear_approximation_theta.txt", theta)


if __name__ == '__main__':
    main()
