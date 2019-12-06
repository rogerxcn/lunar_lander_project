import matplotlib.pyplot as plt


rewards_64 = []
episodes_64 = []
count = 0
rewards_128 = []
episodes_128 = []
rewards_256 = []
episodes_256 = []

with open('./results/dqn_64.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_64.append(float(line[3].strip()))
        episodes_64.append(count)
        count += 1

count=0
with open('./results/dqn_128.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_128.append(float(line[3].strip()))
        episodes_128.append(count)
        count += 1

count=0
with open('./results/dqn_256.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_256.append(float(line[3].strip()))
        episodes_256.append(count)
        count += 1


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Acquired Rewards")
plt.plot(episodes_64, rewards_64, 'r', label='DQN 64 hidden neurons')
plt.plot(episodes_128, rewards_128, 'g', label='DQN 128 hidden neurons')
plt.plot(episodes_256, rewards_256, 'b', label='DQN 256 hidden neurons')
ax.set_xlabel("Iterations")
ax.set_ylabel("Average Reward")
ax.legend(loc='best')
plt.savefig('dqn_plot_engine_failure.jpg')
plt.show()