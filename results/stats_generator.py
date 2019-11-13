import numpy as np
import matplotlib.pyplot as plt

fn1 = "sarsa_reward.txt"
fn2 = "random_reward.txt"
fn3 = "sarsa_reward_1.txt"

x_length = 10000

g1 = np.loadtxt(fn1)
g2 = np.loadtxt(fn2)
g3 = np.loadtxt(fn3)

x = np.linspace(0, x_length, g1.shape[0]-1)

plt.plot(x, g1[1:], label='Sarsa (attempt 1)', color="blue")
plt.plot(x, g3[1:], label='Sarsa (attempt 2)', color="green")
plt.plot(x, g2[1:], label='Random', color="grey")
plt.legend()
plt.title("Acquired Rewards")
plt.xlabel("#Iterations")
plt.ylabel("Reward")

plt.savefig("fusion.png")
