import gym
import gym_custom
import numpy as np
import qlearn
import time
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

env = gym.make('TurtlebotFollowsWalls-v0')

observation_n = env.reset()

#init the network
tf.reset_default_graph()

X = tf.placeholder(shape = [1,3], dtype=tf.float32)
"""
W = tf.Variable(tf.random_uniform([3,8],0, 0))
b = tf.Variable(tf.zeros([1,6]))
L1 = tf.nn.sigmoid(tf.matmul(X, W))
W2 = tf.Variable(tf.random_uniform([128,8], 0, 0))
b2 = tf.Variable(tf.zeros([1,8]))
"""
hidden = slim.fully_connected(X, 6, biases_initializer=None, activation_fn = None)
Qout = slim.fully_connected(hidden,8,activation_fn=None,biases_initializer=None)
predict = tf.argmax(Qout, 1)

#Init loss function and optimizer
Qnext = tf.placeholder(shape = [1,8], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Qnext - Qout))
trainer = tf.train.AdamOptimizer(learning_rate = 0.1)
updateModel = trainer.minimize(loss)

#Training the network
init = tf.global_variables_initializer()

epsilon = 0.72
gamma = 0.8
epsilon_decay = 0.999

save_ep = 1

episode = 0
start_time = time.time()

#reward and step
rList = []
sList = []
with open('step_list_fw_qn.txt', 'r') as f:
    for line in f:
        sList.append(int(line))
with open('reward_list_fw_qn.txt', 'r') as f:
    for line in f:
        rList.append(float(line))
episode = len(sList)

#remember actions
memory_actions = []

#save and restore network
saver = tf.train.Saver()

start_time = time.time()


#train
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "./turtlebotdql.ckpt")
    print("Model restored from file")
    #print sess.run(W)
    while (True):
        episode += 1
        observation = env.reset()
        observation = np.asarray(observation).reshape([1,3])
        step = 0
        total_reward = 0
        if epsilon > 0.05:
            epsilon *= epsilon_decay
        for i in range(5000):
            observation_old = observation
            action, allQ = sess.run([predict, Qout], feed_dict = {X :observation_old})
            if np.random.rand(1) < epsilon:
                action[0] = np.random.randint(8)
            observation, reward, done, _ = env.step(action[0])
            observation = np.asarray(observation).reshape([1,3])
            memory_actions.append([observation_old, action, reward, done, observation])
            total_reward += reward
            if done == True:
                sList.append(i)
                break
            else:
                if i == 4999:
                    sList.append(i)

        indices = np.random.choice(len(memory_actions), min(500, len(memory_actions)))
        for ip in indices:
            observation_old, action, reward, done, observation = memory_actions[ip]
            a, allQ = sess.run([predict, Qout], feed_dict={X: observation_old})
            target = allQ
            #print "op"
            #print allQ
            Q1 = sess.run(Qout, feed_dict = {X: observation})
            maxQ = np.max(Q1)
            
            if done == True:
                target[0, action] = reward
            else:
                target[0, action] = reward + gamma * maxQ
            #print target
            #print sess.run(W)
            #print sess.run(W2)
            _ = sess.run(updateModel, feed_dict = {X: observation_old, Qnext :target})
            a, allQ = sess.run([predict, Qout], feed_dict={X: observation_old})
            #print sess.run(W)
            #print sess.run(W2)
            #print allQ
        memory_actions = []
        rList.append(total_reward)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(episode) + " - epsilon: "+str(round(epsilon,2))+"] - Reward: "+str(total_reward)+"     Time: %d:%02d:%02d" % (h, m, s))
        if episode % save_ep == 0:
            #print sess.run(W)
            save_path = saver.save(sess, "./turtlebotdql.ckpt")
            print("Model saved in file: %s" % save_path)
            with open('step_list_fw_qn.txt', 'w') as f:
                for i in sList:
                    f.write(''.join([str(i),'\n']))
            with open('reward_list_fw_qn.txt', 'w') as f:
                for i in rList:
                    f.write(''.join([str(i),'\n']))

"""
#Init qlearning
qlearn = qlearn.QLearn(actions=range(8),
                    alpha=0.2, gamma=0.8, epsilon=1.0, _file = 'q-table-fw.txt')
initial_epsilon = qlearn.epsilon
epsilon_discount = 0.9986

save_ep = 1

episode = 0
start_time = time.time()

#reward and step
rList = []
sList = []
with open('step_list.txt', 'r') as f:
    for line in f:
        sList.append(int(line))
with open('reward_list.txt', 'r') as f:
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
        with open('step_list.txt', 'w') as f:
            for i in sList:
                f.write(''.join([str(i),'\n']))
        with open('reward_list.txt', 'w') as f:
            for i in rList:
                f.write(''.join([str(i),'\n']))
"""
