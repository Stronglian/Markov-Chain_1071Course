# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:38:17 2018
verify
"""
import numpy as np
#import random
np.set_printoptions(suppress=True)
class HidenMarkovModel_Verify():
    def __init__(self):
        #state
        self.initialStateProb   = np.array([0.6, 0.4]) #[fair, unfair]
        self.stateName = ['fair', 'unfair']
        self.stateNumber = len(self.initialStateProb)
        # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
        self.stateChangeMatrix = np.array([[0.95, 0.05],
                                           [0.10, 0.90]]) 
        #[i][j] 在state i 時，生成j的機率    
        self.probabilityMatrix = np.array([[ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6], #fair 
                                           [1/10, 1/10, 1/10, 1/10, 1/10,  1/2]])#unfair 
        #output
        self.stateOutput = np.array([str(i) for i in range(1, len(self.probabilityMatrix[0,:])+1)])
        #驗證資料正確性
        if (not len(self.initialStateProb) == len(self.stateChangeMatrix)) \
            or (not len(self.initialStateProb) == len(self.probabilityMatrix)) \
            or (not len(self.initialStateProb) == len(self.stateName)):
            raise AssertionError("State 數量未對上")
        if (not len(self.initialStateProb) == len(self.stateChangeMatrix[:,0])): 
            raise AssertionError("輸出 數量未對上")
        return

    def a_prob(self, preState, nextState):
        """(未使用)從 preState 換到 nextState 的機率"""
        prob = self.stateChangeMatrix[preState, nextState]
        return prob
    
    def b_prob(self, state, output):
        """(可以簡化進而不使用)從 state 生出 output 的機率"""
        prob = self.probabilityMatrix[state, self.stateOutput == output]
        return prob
    
    def VerifyProb(self, target, stateSeq, boolPrint=True):
        """ target = "123456"
            stateSeq = "000000"
        """
        #初始化
#        print(self.initialStateProb[int(stateSeq[0])])
#        print(self.probabilityMatrix[int(stateSeq[0]), self.stateOutput == target[0]][0])
#        print(self.b_prob(int(stateSeq[0]), target[0]))
        probOutput = self.initialStateProb[int(stateSeq[0])] * \
                     self.probabilityMatrix[int(stateSeq[0]), self.stateOutput == target[0]][0]
        #
        for i in range(1, len(target)):
#            print(i)
#            print(self.stateChangeMatrix[ int(stateSeq[i-1]), int(stateSeq[i])])
#            print(self.probabilityMatrix[int(stateSeq[i]), self.stateOutput == target[i]][0])
            probOutput *= self.stateChangeMatrix[ int(stateSeq[i-1]), int(stateSeq[i])] * \
                          self.probabilityMatrix[int(stateSeq[i]), self.stateOutput == target[i]][0]
#            print(probOutput)
        if boolPrint:
            print(target, "in", stateSeq, "Prob is:", probOutput)
        return probOutput
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
    test = HidenMarkovModel_Verify()
#    test.VerifyProb("123456" , "000000")
    test.VerifyProb("111666" , "111111")
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  