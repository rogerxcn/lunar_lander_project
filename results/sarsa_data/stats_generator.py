import numpy as np
import matplotlib.pyplot as plt

fn1 = "sarsa_reward_noXY.txt"
fn2 = "sarsa_reward_3XY.txt"
fn3 = "sarsa_reward_5X4Ys.txt"
fn4 = "sarsa_reward_7X5Y.txt"

x_length = 10000

g1 = np.loadtxt(fn1)
g2 = np.loadtxt(fn2)
g3 = np.loadtxt(fn3)
g4 = np.loadtxt(fn4)

x = np.linspace(0, x_length, g1.shape[0]-1)

plt.plot(x, g1[1:], label='1X1Y', color="green")
plt.plot(x, g2[1:], label='3X1Y', color="blue")
plt.plot(x, g3[1:], label='5X4Y', color="red")
plt.plot(x, g4[1:], label='7X5Y', color="purple")

plt.legend()
plt.title("Comparison of State Discretization")
plt.xlabel("Iterations")
plt.ylabel("Average Reward")

plt.savefig("sarsa_discretization.png")
