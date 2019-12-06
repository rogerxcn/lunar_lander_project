import numpy as np
import matplotlib.pyplot as plt

fn1 = "noisy_pomdp_agent.txt"
fn2 = "noisy_agent.txt"
fn3 = "noisy_sarsa_2.txt"

x_length = 1000

g1 = np.loadtxt(fn1)
g2 = np.loadtxt(fn2)
g3 = np.loadtxt(fn3)

x = np.linspace(0, x_length, g1.shape[0]-1)

plt.plot(x, g1[1:], label='POMDP', color="green", alpha=0.4)
plt.plot([0, x_length], [np.mean(g1), np.mean(g1)], '--', color="green")
plt.plot(x, g2[1:], label='Vanilla Q', color="blue", alpha=0.4)
plt.plot([0, x_length], [np.mean(g2), np.mean(g2)], '--', color="blue")
plt.plot(x, g3[1:], label='Trained Q', color="orange", alpha=0.4)
plt.plot([0, x_length], [np.mean(g3), np.mean(g3)], '--', color="orange")

plt.legend()
plt.title("Comparison Under Noisy Observation")
plt.xlabel("Iterations")
plt.ylabel("Average Reward")

plt.savefig("noisy_obs.png")
