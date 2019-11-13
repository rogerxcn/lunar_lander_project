import numpy as np
import matplotlib.pyplot as plt

fn1 = "sarsa_reward.txt"
fn2 = "random_reward.txt"
fn4 = "naive_sarsa_reward.txt"

x_length = 10000

g1 = np.loadtxt(fn1)
g2 = np.loadtxt(fn2)
g4 = np.loadtxt(fn4)

x = np.linspace(0, x_length, g1.shape[0]-1)

plt.plot(x, g1[1:], label='Sarsa (optimized)', color="green")
plt.plot(x, g4[1:], label='Sarsa (naive)', color="blue")
plt.plot(x, g2[1:], label='Random', color="grey")
plt.legend()
plt.title("Acquired Rewards")
plt.xlabel("#Iterations")
plt.ylabel("Reward")

plt.savefig("fusion.png")
