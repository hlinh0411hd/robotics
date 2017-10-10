import numpy as np
import tensorflow as tf
import gym
import gym_custom
import sys
import os
import time

env = gym.make('TurtlebotWorld-v0')

save_episode = 1

#Implementing the network

tf.reset_default_graph()


#Establish the feed-forward part of network used to choose action
X = tf.placeholder(shape = [1,3], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([3,3], 0, 0.01))
Qout = tf.matmul(X, W)
predict = tf.argmax(Qout,1)


#Obtain the loss 
nextQ = tf.placeholder(shape = [1,3], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.AdamOptimizer(learning_rate = 0.2)
updateModel = trainer.minimize(loss)

#Training the network
init = tf.global_variables_initializer()

epsilon = 0.99
gamma = 0.8
epsilon_decay = 0.9986

#lists contain total rewards and steps per episodes
sList = []
rList = []

episode = 0

#save and restore network
saver = tf.train.Saver()

start_time = time.time()

with tf.Session() as sess:
    sess.run(init)
    while (True):
        episode += 1
        observation = env.reset()
        observation = np.asarray(observation).reshape([1,3])
        step = 0
        total_reward = 0
        while (step <= 200000):
            step += 1
            #get action and run
            action, allQ = sess.run([predict, Qout], feed_dict = {X:observation})
            print observation
            print allQ
            print sess.run(W)
            action = action[0]
            if np.random.rand(1) < epsilon:
                action = np.random.randint(3)
            
            observation_old = observation
            observation, reward, done, _ = env.step(action)
            observation = np.asarray(observation).reshape([1,3])
            
            #get Q value of new observation
            Q1 = sess.run(Qout, feed_dict = {X:observation})
           
            #get max of Q value of new observation and update to old observation
            maxQ = np.max(Q1)
            targetQ = allQ
            targetQ[0, action] = reward + gamma * maxQ
            print targetQ

            #train Q network
            _, W1 = sess.run([updateModel,W], feed_dict = {X:observation_old, nextQ: targetQ})
            total_reward += reward

            if done:
                break
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(episode+1)+" - epsilon: "+str(round(epsilon,2))+"] - Reward: "+str(total_reward)+"     Time: %d:%02d:%02d" % (h, m, s))
        print sess.run(W)
        sList.append(step)
        rList.append(total_reward)
        epsilon *= epsilon_decay
        

        if episode % save_episode ==0 :
            #print sList and rList
            with open('step_list.txt', 'w') as f:
                for i in sList:
                    f.write(''.join([str(i),'\n']))
            with open('reward_list.txt', 'w') as f:
                for i in rList:
                    f.write(''.join([str(i),'\n']))
            
            save_path = saver.save(sess, "./turtlebotdql.ckpt")
            print("Model saved in file: %s" % save_path)



