import numpy as np
import random

class Agent():
    crouching = (47, 124, 57, 28)
    standing = (45, 107, 42, 45)
    position = (70, 127)
    name = 'runner'
    nactions = 3

class Qlearn():

    def __init__(self, nstates):
        self.q_table = np.zeros([nstates,Agent.nactions])
        #self.q_table = self.q_table/(np.max(self.q_table)*100-1)
        self.indexes = {}
        self.cont_ind = 0
        self.cont = 0

        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.002
    
    def getAction(self, state):

        try:
            self.indexes[state]
        except:
            self.indexes[state] = self.cont_ind
            self.cont_ind+=1
        if (state[1] == 1):
            return 2
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0,Agent.nactions-1)
        else :
            return np.argmax(self.q_table[self.indexes[state]])

    def getMax(self, state):
        try:
            self.indexes[state]
        except:
            self.indexes[state] = self.cont_ind
            self.cont_ind+=1
        return np.max(self.q_table[self.indexes[state]])
    
    def get_qvalue(self, state, action):
        try:
            self.indexes[state]
        except:
            self.indexes[state] = self.cont_ind
            self.cont_ind+=1
        return self.q_table[self.indexes[state]][action]

    def set_new_qvalue(self, old_value, reward, next_max, state, action):
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        #print('s:', state,'a', action, 'r:',reward, 'o',old_value,'n', next_max)
        self.q_table[self.indexes[state]][action] = new_value