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
            print("State 數量未對上")
            raise AssertionError
        if (not len(self.initialStateProb) == len(self.stateChangeMatrix[:,0])): 
            print("輸出 數量未對上")
            raise AssertionError
        return
    def CalRoPsiTable(self, target):
        """ """
        T = len(target)
        #step 0 - table #table - time(-1) - state #ρ (ro)、ψ(psi)
        ro  = np.zeros((len(target), self.stateNumber), dtype = np.float32)
        psi = np.zeros((len(target), self.stateNumber), dtype = np.float32)
        #Step 1 – Initialization #psi DONE before
        ro[0,:] = self.initialStateProb * self.probabilityMatrix[:, self.stateOutput == target[0]].T
        psi[0,:] = np.array([ -1 for i in range(self.stateNumber)])#0 重複意義，故換-1
        #Step 2 – Recursion
        for t in range(1, len(target)):
            for s in range(self.stateNumber):
                ro[t, s]  = (ro[t-1, s] * self.stateChangeMatrix[:, s]).max() *     \
                            self.probabilityMatrix[s, self.stateOutput == target[t]]
                psi[t, s] = (ro[t-1, s] * self.stateChangeMatrix[:, s]).argmax() #*  \
#                            self.probabilityMatrix[s, self.stateOutput == target[t]] #不用算入，因為乘了相對大小還是不變
        #Step 3 – Termination
#        P_all  = ro[:,:].max(axis = 1)    #P*
#        iT_all = ro[:,:].argmax(axis = 1) #i*
        P_all  = ro[-1,:].max()    #P*
        iT_all = ro[-1,:].argmax() #i*
#        print(P_all, iT_all)
        #Step 4 – Path (state sequence) backtracking
        stateSeqIndex = np.array([-1 for i in range(T)])
        stateSeqIndex[-1] = iT_all
        for t in range(len(target)-1, 0, -1):
#        for t, index in list(enumerate(reversed(iT_all)))[1:]:
#            print(stateSeqIndex[t])
            stateSeqIndex[t-1] = psi[t, stateSeqIndex[t]]
#            iT_all[t-1] = psi[t, index]
#        assert True == False
        #另外轉存
        self.roTable = ro
        self.psiTable = psi
        return ro, psi, P_all, stateSeqIndex
    def Predict_optimalStateSequence_useRoPsi(self,target):
        """ """
        ro, psi, bestProb, bestStateSquenceIndex = self.CalRoPsiTable(target)
        #The best state sequence having the highest probability
        print('"',target,'"',"'s Probability:", bestProb)#,beta[0,:].sum(dtype = np.float32))
        seqLis = []
        for indexTmp in bestStateSquenceIndex: #ro[:,:].argmax(axis = 1):
            seqLis.append(self.stateName[indexTmp])
        print('"',target,'"',"'s Optimal State Sequence is", seqLis)
        return
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
    
    test = HidenMarkovModel_Viterbi()
    test.Predict_optimalStateSequence_useRoPsi(target = "123456")
    ro, psi = test.roTable, test.psiTable
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  

