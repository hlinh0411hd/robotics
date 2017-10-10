import gym
import gym_custom
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import os
import time
import random

env = gym.make('TurtlebotAroundCamera-v0')

observation = env.reset


#init the network
class Qnetwork():
    def __init__(self):
        self.X = tf.placeholder(shape = (120, 160), dtype = tf.float32)
        self._X = tf.reshape(self.X, [-1, 120, 160, 1])
        self.W_c1 = self.weight_variable([5, 5, 1, 32])
        self.b_c1 = self.bias_variable([32])
        self.conv = tf.nn.relu(self.conv2d(self._X, self.W_c1) + self.b_c1)
        """self.conv1 = self.max_pool(tf.nn.relu(self.conv2d(self._X, self.W_c1) + self.b_c1), [1, 2, 2, 1], [1, 2, 2, 1])
        self.W_c2 = self.weight_variable([5, 5, 32, 32])
        self.b_c2 = self.bias_variable([32])
        self.conv2 = self.max_pool(tf.nn.relu(self.conv2d(self.conv1, self.W_c2) + self.b_c2), [1, 2, 2, 1], [1, 2, 2, 1])
        self.W_c3 = self.weight_variable([5, 5, 32, 64])
        self.b_c3 = self.bias_variable([64])
        self.conv3 = self.max_pool(tf.nn.relu(self.conv2d(self.conv2, self.W_c3) + self.b_c3), [1, 2, 2, 1], [1, 2, 2, 1])
        self.conv3_flat = tf.reshape(self.conv3, [-1, 15 * 20 * 64])
        self.W_fc1 = self.weight_variable([15 * 20 * 64, 256])
        self.b_fc1 = self.bias_variable([256])
        self.fc1 = tf.nn.relu(tf.matmul(self.conv3_flat, self.W_fc1) + self.b_fc1)
        self.W_fc2 = self.weight_variable([256, 5])
        self.b_fc2 = self.bias_variable([5])
        """
        self.conv_flat = tf.reshape(self.conv, [-1, 120 * 160 * 32])
        self.W_fc = self.weight_variable([120 * 160 * 32, 5])
        self.b_fc = self.bias_variable([5])
        self.Qout = tf.matmul(self.conv_flat, self.W_fc) + self.b_fc
        self.predict = tf.argmax(self.Qout, 1)

        self.nextQ = tf.placeholder(shape = [1, 5], dtype = tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Qout - self.nextQ))
        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        
        
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")
    
    def max_pool(self, x, _ksize, _strides):
        return tf.nn.max_pool(x, ksize = _ksize, strides = _strides, padding = "SAME")
    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

class ExperienceReplay():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
 
    def add(self, experience):
        if len(self.buffer) + len(experience) > self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, min(size, len(self.buffer)))), [min(size, len(self.buffer)), 5])


#set parameter 
batch_size = 50
gamma = 0.8
epsilon = 1.0
epsilon_decay = 0.9/3000
load_model = False
alpha = 0.2
path = "./arounddql"
pre_train_step = 5000

save_ep = 50

tf.reset_default_graph()
mainQN = Qnetwork()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
replay = ExperienceReplay()

rList = []
sList = []
episode = 0
total_step = 0

if not os.path.exists(path):
    os.makedirs(path)
start_time = time.time()
with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('loading model....')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        with open(path + '/step_list_ta_dql.txt', 'r') as f:
            for line in f:
                sList.append(int(line))
        with open(path + '/reward_list_ta_dql.txt', 'r') as f:
            for line in f:
                rList.append(float(line))
        episode = len(sList)
    while True:
        episode += 1
        observation = env.reset()
        total_reward = 0
        total_random = 0
        replay_ep = ExperienceReplay()
        for i in range(5000):
            if np.random.rand(1) < epsilon or total_step < pre_train_step:
                action = np.random.randint(5)
                total_random += 1
            else:
                action = sess.run(mainQN.predict, feed_dict = {mainQN.X : observation})[0]

            observation_old = observation
            observation, reward, done, _ = env.step(action)
            if observation == []:
                break
            replay_ep.add(np.reshape([observation_old, action, reward, done, observation], [1, 5]))
            total_step += 1
            
            total_reward += reward
            if done or i == 4999:
                sList.append(i+1)
            if done:
                break
            
            if total_step > pre_train_step:
                if total_step % 4 == 0:
                    trainBatch = replay.sample(batch_size)
                    for tb in trainBatch:
                        observation_old, action, reward, done, observation = tb
                        
                        Qall = sess.run(mainQN.Qout, feed_dict = {mainQN.X : observation_old})

                        Q1 = sess.run(mainQN.Qout, feed_dict = {mainQN.X : observation})
                        maxQ = np.max(Q1, 1)
                        target = Qall
                        if done:
                            target[0, action] = reward
                        else:
                            target[0, action] = reward + gamma * maxQ
                        
                        _ = sess.run(mainQN.updateModel, feed_dict = {mainQN.X : observation_old, mainQN.nextQ : target})

                    
        if total_step > pre_train_step:
            if epsilon > 0.1:
                epsilon -= epsilon_decay

        replay.add(replay_ep.buffer)
        rList.append(total_reward)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(episode) + " - epsilon: "+str(round(epsilon,2))+"] - Random: "+ str(total_random)+"] - Reward: "+str(total_reward)+" Step: " + str(sList[-1]) + "     Time: %d:%02d:%02d" % (h, m, s))
        if episode % save_ep == 0:
            #print sess.run(W)
            saver.save(sess,path+'/model_depth.ckpt')
            print("Saved Model")
            with open(path + '/step_list_ta_dql.txt', 'w') as f:
                for i in sList:
                    f.write(''.join([str(i),'\n']))
            with open(path + '/reward_list_ta_dql.txt', 'w') as f:
                for i in rList:
                    f.write(''.join([str(i),'\n']))


    
