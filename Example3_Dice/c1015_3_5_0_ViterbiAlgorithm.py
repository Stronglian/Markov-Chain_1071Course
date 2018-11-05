# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:37:23 2018
"""
import numpy as np
#import random
np.set_printoptions(suppress=True)
class HidenMarkovModel_Viterbi():
    def __init__(self):
        #state
        self.initialStateProb   = np.array([0.6, 0.4]) #[fair, unfair] #為了統一就不改直的了
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
    
#    def a_prob(self, preState, nextState):
#        """(未使用)從 preState 換到 nextState 的機率"""
#        prob = self.stateChangeMatrix[preState, nextState]
#        return prob
#    
#    def b_prob(self, state, output):
#        """(可以簡化進而不使用)從 state 生出 output 的機率"""
#        prob = self.probabilityMatrix[state, self.stateOutput == output]
#        return prob
        
    def CalRoPsiTable(self, target):
        """ """
        lenTarget = len(target)
        # step 0 - table #table - time(-1) - state #ρ (ro)、ψ(psi)
        ro  = np.zeros((len(target), self.stateNumber), dtype = np.float32)
        psi = np.ones((len(target), self.stateNumber), dtype = np.float32) * -1
        # step 1 – Initialization #psi DONE before
        ro[0,:]  = self.initialStateProb * self.probabilityMatrix[:, self.stateOutput == target[0]].T[0]
#        print(self.initialStateProb ,"\n",  self.probabilityMatrix[:, self.stateOutput == target[0]].T[0], "\n")
#        psi[0,:] = np.array([ -1 for i in range(self.stateNumber)])#0 重複意義，故換-1
        # step 2 – Recursion
        tmpArr = np.zeros((2,2))
        for t in range(1, len(target)):
#            for s in range(self.stateNumber):
#                tmpArr[:, s] = ro[t-1, s] * self.stateChangeMatrix[:, s] * self.probabilityMatrix[s, self.stateOutput == target[t]]
            
#                print((ro[t-1, s] * self.stateChangeMatrix[:, s]).max())# *     \
#            print(( (ro[t-1, :] * self.stateChangeMatrix[:, :])*     \
#                            self.probabilityMatrix[:, self.stateOutput == target[t]]).max(axis = 1))
#                psi[t, s] = (ro[t-1, s] * self.stateChangeMatrix[:, s]).argmax() #*  \
#                            self.probabilityMatrix[s, self.stateOutput == target[t]] #不用算入，因為乘了相對大小還是不變

#            print("ro", ro[t-1, :] , "Change", self.stateChangeMatrix[:, :], 
#                  "Prob", self.probabilityMatrix[:, self.stateOutput == target[t]].T[0], sep = "\n")
            tmpArr[:, :] = ro[t-1, :] * self.stateChangeMatrix[:, :] * self.probabilityMatrix[:, self.stateOutput == target[t]].T[0]
            ro[t, :]  = tmpArr.max(axis = 0)
            psi[t, :] = tmpArr.argmax(axis = 0)
            
        self.roTable = ro.copy()
        self.psiTable = psi.copy()
        # step 3 – Termination
        P_star_all = ro[-1,:].max().copy()    #P*
        iT_all     = ro[-1,:].argmax().copy() #i*
#        print(P_all, iT_all)
        # step 4 – Path (state sequence) backtracking
        stateSeqIndex = np.array([-1 for i in range(lenTarget)])
#        print(stateSeqIndex)
        stateSeqIndex[-1] = iT_all
        for t in range(len(target)-1, 0, -1):
#        for t, index in list(enumerate(reversed(iT_all)))[1:]:
#            print(stateSeqIndex)
            stateSeqIndex[t-1] = psi[t, stateSeqIndex[t]]
#        print(stateSeqIndex)
        #另外轉存
#        self.roTable = ro.copy()
#        self.psiTable = psi.copy()
        return ro, psi, P_star_all, stateSeqIndex
    
    def Predict_optimalStateSequence_useRoPsi(self,target, boolPrint = True):
        """ """
        ro, psi, bestProb, bestStateSquenceIndex = self.CalRoPsiTable(target)
        #The best state sequence having the highest probability
        if boolPrint:
            print('"',target,'"',"'s Probability:", bestProb) #,beta[0,:].sum(dtype = np.float32))
        seqLis = []
        for indexTmp in bestStateSquenceIndex: #ro[:,:].argmax(axis = 1):
            seqLis.append(self.stateName[indexTmp])
        if boolPrint:
            print('"',target,'"',"'s Optimal State Sequence is", seqLis)
        return bestProb, bestStateSquenceIndex
    
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
    
    test = HidenMarkovModel_Viterbi()
    
#    test.Predict_optimalStateSequence_useRoPsi(target = "666")
#    test.Predict_optimalStateSequence_useRoPsi(target = "123456")
#    test.Predict_optimalStateSequence_useRoPsi(target = "664321")# state seq!
#    test.Predict_optimalStateSequence_useRoPsi(target = "111666")
#    test.Predict_optimalStateSequence_useRoPsi(target = "162636")
#    test.Predict_optimalStateSequence_useRoPsi(target = "126656")
    test.Predict_optimalStateSequence_useRoPsi(target = "1"*6+"6"*6)
    ro, psi = test.roTable, test.psiTable
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  

