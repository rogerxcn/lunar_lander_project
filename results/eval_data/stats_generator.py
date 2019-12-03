import numpy as np
import matplotlib.pyplot as plt

fn1 = "noisy_pomdp_agent.txt"
fn2 = "noisy_agent.txt"

x_length = 1000

g1 = np.loadtxt(fn1)
g2 = np.loadtxt(fn2)

x = np.linspace(0, x_length, g1.shape[0]-1)

plt.plot(x, g1[1:], label='POMDP', color="green", alpha=0.4)
plt.plot([0, x_length], [np.mean(g1), np.mean(g1)], '--', color="green")
plt.plot(x, g2[1:], label='Vanilla', color="blue", alpha=0.4)
plt.plot([0, x_length], [np.mean(g2), np.mean(g2)], '--', color="blue")

plt.legend()
plt.title("POMDP Agent vs Vanilla Agent")
plt.xlabel("Iterations")
plt.ylabel("Average Reward")

plt.savefig("fusion.png")
