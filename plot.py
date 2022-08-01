import matplotlib.pyplot as plt
import random

step = 0
reward = 0
fig = plt.figure()
ax = fig.add_subplot(111)

steps, cumulative_reward = [], []
while step < 100000:
    steps.append(step)
    cumulative_reward.append(reward)
    
    ax.plot(steps, cumulative_reward, color='b')
    fig.canvas.draw()
    fig.show()
    
    step += 1
    reward += random.randint(-100, 100)
    plt.pause(0.001)

