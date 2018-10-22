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
class HidenMarkovModel_foward():
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
    def CalAlphaTable(self, target):
        """ alpha 指定生成數列、時間t時，在 state j 的機率
        alpha: (指定輸出下，)第t次，輸出指定序列之機率。
        a: 狀態轉換機率；self.stateChangeMatrix[preState, nextState]
        b: 該狀態輸出該物機率；self.probabilityMatrix[state, self.stateOutput == output]
        """
        #alpha - time(-1) - state
        alpha = np.zeros((len(target), self.stateNumber), dtype = np.float32)
        
        #初始 t = 0，生成第一個target的機率
#        for j, pi in enumerate(self.initialStateProb):
#            alpha[0, j] = pi * self.b_prob(j, target[0])
        alpha[0, :] = self.initialStateProb * self.probabilityMatrix[:, self.stateOutput == target[0]].T
#        alpha[0, :] = np.multiply(self.initialStateProb , self.probabilityMatrix[:, self.stateOutput == target[0]].T)
        
        #剩下的字串
        for t in range(1, len(target)):
            #合運算 - 1
#            for j in range(self.stateNumber):
#                tempSum = sum([ alpha[t-1,s]*self.a_prob(s, j)  for s in range(self.stateNumber)])
#                tempSum = np.multiply(alpha[t-1,:], self.stateChangeMatrix[:, j]).sum(dtype = np.float32)
#                #print(tempSum, end = '')
#                #print(tempSum)
#                alpha[t, j] = tempSum * self.probabilityMatrix[j, self.stateOutput == target[t]] #self.b_prob(j, target[t])
#            #print()
            #合運算 - 2
            tempSum = np.multiply(alpha[t-1,:], self.stateChangeMatrix.T).sum(axis = 1, dtype = np.float32)
            #print(tempSum)
            alpha[t, :] = np.multiply(tempSum, self.probabilityMatrix[:, self.stateOutput == target[t]].T)#self.b_prob(j, target[t])
        return alpha
    
    def PredictUseAlpha(self, target):
        """ 將Alpha最後兩組相加，便是所求"""
        alpha = self.CalAlphaTable(target)
        print('alphaTable:\n',alpha)
        print('"',target,'"',"'s Probability:", alpha[-1,:].sum(dtype = np.float32))
        return
    
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
    
    test = HidenMarkovModel_foward()
    test.PredictUseAlpha(target = "123456")
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  

