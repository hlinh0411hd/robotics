import gym
import gym_custom
import numpy as np
import qlearn
import time
import random

env = gym.make('TurtlebotFollowsWalls-v0')

observation_n = env.reset()
"""
print observation_n

for i in range(10):
    observation = env.reset()
    for j in range(10):
        env.render()
        action = np.random.randint(4)
        observation, reward, done, info = env.step(action)
        print observation
"""

#Init qlearning
qlearn = qlearn.QLearn(actions=range(8),
                    alpha=0.2, gamma=0.8, epsilon=1.0, _file = 'q-table-fw1.txt')
initial_epsilon = qlearn.epsilon
epsilon_discount = 0.9986

save_ep = 1

episode = 0
start_time = time.time()

#reward and step
rList = []
sList = []
with open('step_list_fw1.txt', 'r') as f:
    for line in f:
        sList.append(int(line))
with open('reward_list_fw1.txt', 'r') as f:
    for line in f:
        rList.append(float(line))
episode = len(sList)
while (True):
    episode += 1
    observation = env.reset()
    total_reward = 0
    if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
    state = ''
    for i in observation:
        state = state + str(i)
    #state = observation
    for i in range(5000):

        # Pick an action based on the current state
        action = qlearn.chooseAction(state)

        # Execute the action and get feedback
        observation, reward, done, info = env.step(action)
        total_reward += reward
        nextState = ''
        for j in observation:
            nextState = nextState + str(j)
        #print nextState
        #print '\n'
        #nextState = observation
        qlearn.learn(state, action, reward, nextState)

        if not(done):
            state = nextState
            if i == 4999:
                sList.append(i)
                break
        else:
            sList.append(i)
            break 
    rList.append(total_reward)
    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)
    print ("EP: "+str(episode+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(total_reward)+"     Time: %d:%02d:%02d" % (h, m, s))
    if episode % save_ep == 0:
        qlearn.save()
        with open('step_list_fw1.txt', 'w') as f:
            for i in sList:
                f.write(''.join([str(i),'\n']))
        with open('reward_list_fw1.txt', 'w') as f:
            for i in rList:
                f.write(''.join([str(i),'\n']))

