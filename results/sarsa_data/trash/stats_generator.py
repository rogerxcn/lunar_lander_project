import numpy as np
import matplotlib.pyplot as plt

fn1 = "naive_sarsa_reward.txt"
fn2 = "random_reward.txt"

x_length = 1000

g1 = np.loadtxt(fn1)
g2 = np.loadtxt(fn2)

x = np.linspace(0, x_length, g1.shape[0]-1)

plt.plot(x, g1[1:], label='Naive Sarsa', color="green", alpha=1)
plt.plot([0, x_length], [np.mean(g1), np.mean(g1)], '--', color="green", alpha=0.4)
plt.plot(x, g2[1:], label='Random', color="blue", alpha=1)
plt.plot([0, x_length], [np.mean(g2), np.mean(g2)], '--', color="blue", alpha=0.4)

plt.legend()
plt.title("Naive Sarsa Agent vs Random Agent")
plt.xlabel("Iterations")
plt.ylabel("Average Reward")

plt.savefig("naive_vs_random.png")
