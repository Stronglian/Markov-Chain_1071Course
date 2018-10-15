# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 00:04:28 2018

@author: StrongPria
"""
"""
待處理:
    輸出與 array index 的對應問題
    矩陣運算
"""
import numpy as np
#import random
np.set_printoptions(suppress=True)
class HidenMarkovModel_backward():
    def __init__(self):
        self.initialStateProb   = np.array([0.6, 0.4]) #[fair, unfair]
        # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
        self.stateChangeMatrix = np.array([[0.95, 0.05],
                                           [0.10, 0.90]]) 
        #[i][j] 在state i 時，生成j的機率    
        self.probabilityMatrix = np.array([[ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6], #fair 
                                           [1/10, 1/10, 1/10, 1/10, 1/10,  1/2]])#unfair 
        self.stateOutput = np.array([str(i) for i in range(1, len(self.probabilityMatrix[0,:])+1)]) 
#        self.stateOutput = np.array([str(i) for i in range(1, 6+1)])
        #驗證資料正確性
        if (not len(self.initialStateProb) == len(self.stateChangeMatrix)) \
            or (not len(self.initialStateProb) == len(self.probabilityMatrix)):
            print("State 數量未對上")
            raise AssertionError
        if (not len(self.initialStateProb) == len(self.stateChangeMatrix[:,0])): 
            print("輸出 數量未對上")
            raise AssertionError
        self.stateNumber = len(self.initialStateProb)
        return

#    def a_prob(self, preState, nextState):
#        """(未使用)從 preState 換到 nextState 的機率"""
#        prob = self.stateChangeMatrix[preState, nextState]
#        return prob
#    
#    def b_prob(self, state, output):
#        """(可以簡化進而不使用)從 state 生出 output 的機率"""
#        prob = self.probabilityMatrix[state, self.stateOutput == output]
#        return prob
#
    def CalBetaTable(self, target):
        """ beta 指定生成數列、時間t時，在 state j 的機率
        alpha: (指定輸出下，)第t次，輸出指定序列之機率。
        beta: 
        a: 狀態轉換機率；self.stateChangeMatrix[preState, nextState]
        b: 該狀態輸出該物機率；self.probabilityMatrix[state, self.stateOutput == output]
        """
        #beta - time(-1) - state
        beta = np.zeros((len(target), self.stateNumber), dtype = np.float32)
        #初始 t = T
        beta[-1, :] = np.ones(len(beta[-1, :]))
        print(beta[-1,:])
        #剩下的字串
        for t in range(len(target)-1, -1, -1): #(t-1) to t
            print('t=',t+1, '===')
            for s in range(self.stateNumber):
                print('t=',t+1, 's=',s)
                a_prob = self.stateChangeMatrix[s, :]
                b_prob = self.probabilityMatrix[:, self.stateOutput == target[t]].T[0]
                print('a_prob',a_prob,'\nb_prob', b_prob,'\nbeta[t, s]',beta[t, :])
                beta[t-1, s] = np.multiply(np.multiply( a_prob, b_prob), beta[t, :]).sum()
            print('beta[t-1, :]', beta[t-1, :], '\n\n\n')
#                beta[t, s] = np.multiply(self.stateChangeMatrix.T[1], self.probabilityMatrix[:, self.stateOutput == target[t]].T, beta[t+1,:])#self.b_prob(j, target[t])
#            print(beta[t, :])
#            assert True == False
#            #合運算 - 2
#            tempSum = np.multiply(beta[t+1,:], self.stateChangeMatrix.T)#.sum(axis = 1, dtype = np.float32)
#            print(tempSum)
#            assert True == False
#            #print(tempSum)
#            beta[t, :] = np.multiply(tempSum, self.probabilityMatrix[:, self.stateOutput == target[t]].T)#self.b_prob(j, target[t])
        #初始 t = 0，生成第一個target的機率
        beta[0, :] = beta[0, :] * self.initialStateProb * self.probabilityMatrix[:, self.stateOutput == target[0]].T
        return beta
    
    def PredictUseBeta(self, target):
        """ 將Beta最後兩組相加，便是所求"""
        beta = self.CalBetaTable(target)
        print('betaTable:\n',beta)
        print('"',target,'"',"'s Probability:",beta[0,:].sum(dtype = np.float32))
        return
    
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
    
    test = HidenMarkovModel_backward()
    test.PredictUseBeta(target = "123456")
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  

