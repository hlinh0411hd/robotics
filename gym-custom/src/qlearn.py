import random
import numpy as np

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma, _file):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        self._file = _file
        with open(self._file, 'r') as f:
            self.epsilon = float(next(f))
            self.alpha = float(next(f))
            self.gamma = float(next(f))
            for line in f:
                state, action, qv = line.split()
                print state, action, qv
                action = int(action)
                qv = float(qv)
                self.q[(state, action)] = qv

    def save(self):
        with open(self._file, 'w') as f:
            f.write(''.join([str(self.epsilon),'\n']))
            f.write(''.join([str(self.alpha),'\n']))
            f.write(''.join([str(self.gamma),'\n']))
            for state, action in self.q:
                if type(state) is str and state.find("False") == -1 and state.find("True") == -1:
                    f.write(' '.join([state, str(action), str(self.q[(state, action)])]))
                    f.write('\n')
        
            

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        """
        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)
        """
        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)
        if random.random() < self.epsilon:
            i = np.random.randint(8)
        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
