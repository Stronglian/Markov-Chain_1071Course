# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:12:47 2018

111666 怪怪的

"""
import numpy as np
#import random
np.set_printoptions(suppress=True)
class HidenMarkovModel_gamma():
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
        """ alpha 指定生成數列、時間t時，在 state j 的機率 #前綴
        alpha: (指定輸出下，)第t次，輸出指定序列之機率。
        a: 狀態轉換機率；self.stateChangeMatrix[preState, nextState]
        b: 該狀態輸出該物機率；self.probabilityMatrix[state, self.stateOutput == output]
        """
        #alpha - time(-1) - state
        alpha = np.zeros((len(target), self.stateNumber), dtype = np.float32)
        #初始 t = 0，生成第一個target的機率
        alpha[0, :] = self.initialStateProb * self.probabilityMatrix[:, self.stateOutput == target[0]].T
        #剩下的字串
        for t in range(1, len(target)):
            tempSum = np.multiply(alpha[t-1,:], self.stateChangeMatrix.T).sum(axis = 1, dtype = np.float32)
            #print(tempSum)
            alpha[t, :] = np.multiply(tempSum, self.probabilityMatrix[:, self.stateOutput == target[t]].T)#self.b_prob(j, target[t])
        #另外轉存
        self.alphaTable = alpha
        return alpha
    
    def PredictUseAlpha(self, target ,boolPrint = True):
        """ 將 Alpha 最後兩組相加，便是所求"""
        alpha = self.CalAlphaTable(target)
        prob_ProO_alpha = alpha[-1,:].sum(dtype = np.float32)
        if boolPrint:
            print('alphaTable:\n',alpha)
            print('"',target,'"',"'s Probability:", prob_ProO_alpha)
        return prob_ProO_alpha
    
    def CalBetaTable(self, target):
        """ beta 指定生成數列、時間t時，在 state j 的機率 #後綴
        beta: (指定輸出下，)第t次之後，輸出指定序列之機率。
        a: 狀態轉換機率；self.stateChangeMatrix[preState, nextState]
        b: 該狀態輸出該物機率；self.probabilityMatrix[state, self.stateOutput == output]
        """
        #beta - time(-1) - state
        beta = np.zeros((len(target), self.stateNumber), dtype = np.float32)
        #初始 t = T
        beta[-1, :] = np.ones(len(beta[-1, :]))
#        print(beta[-1,:])
        #剩下的字串
        for t in range(len(target)-1, 0, -1): #(t-1) to t
#            print('t=',t+1, '===')
            for s in range(self.stateNumber):
#                print('t=',t+1, 's=',s)
                a_prob = self.stateChangeMatrix[s, :]
                b_prob = self.probabilityMatrix[:, self.stateOutput == target[t]].T[0]
#                print('a_prob',a_prob,'\nb_prob', b_prob,'\nbeta['+str(t)+','+str(s)+']',beta[t, :])
                beta[t-1, s] = np.multiply(np.multiply( a_prob, b_prob), beta[t, :]).sum()
#            print('beta['+str(t-1)+', :]', beta[t-1, :], '\n\n\n')
#            assert True == False
        #收尾 t = 0，生成第一個target的機率
#        beta[0, :] = beta[0, :] * self.initialStateProb * self.probabilityMatrix[:, self.stateOutput == target[0]].T
#        beta[0, :] = beta[0, :] 
        #另外轉存
        self.betaTable = beta
        return beta
    
    def PredictUseBeta(self, target, boolPrint=True):
        """ 將 Beta 最前兩組相加，便是所求"""
        beta = self.CalBetaTable(target)
        prob_PrO_bata = (beta[0,:] * self.initialStateProb * self.probabilityMatrix[:, self.stateOutput == target[0]].T).sum(dtype = np.float32)
        if boolPrint:
            print('betaTable:\n',beta)
            print('"',target,'"',"'s Probability:", prob_PrO_bata)
        return prob_PrO_bata
    
    def CalGammaTable(self, target):
        """ 利用 alpha、beta 來算該 state 發生機率，進而推導最佳 State 順序"""
#        alpha = self.CalAlphaTable(target)
#        beta = self.CalBetaTable(target)
        prob_PrO = self.PredictUseAlpha(target, boolPrint=False)
        prob_PrO_beta = self.PredictUseBeta(target, boolPrint=False)
        alpha = self.alphaTable
        beta = self.betaTable
#        prob_PrO = beta[0,:].sum(dtype = np.float32) #alpha[-1,:].sum(dtype = np.float32)
        if not prob_PrO == prob_PrO_beta:
#            raise AssertionError("alpha(",prob_PrO,"),beta(",prob_PrO_beta,")結果不同")
            print("alpha(",prob_PrO,"),beta(",prob_PrO_beta,")結果不同")
        #gamma - time(-1) - state
        gamma = np.zeros((len(target), self.stateNumber), dtype = np.float32)
        gamma = (alpha * beta) / prob_PrO
        #另外轉存
        self.gammaTable = gamma
        return gamma
    
    def Predict_optimalStateSequence_useGamma(self, target):
        """ """
        gamma = self.CalGammaTable(target)
        seqLis = []
#        for t in range(len(target)):
#            indexTmp = gamma[t, :].argmax()
#            seqLis.append(self.stateName[indexTmp])
        for indexTmp in gamma.argmax(axis = 1):
            seqLis.append(self.stateName[indexTmp])
        print('"',target,'"',"'s Optimal State Sequence is", seqLis)
        return
    
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
    
    test = HidenMarkovModel_gamma()
    test.Predict_optimalStateSequence_useGamma(target = "111666")
    alpha, beta, gamma = test.alphaTable, test.betaTable, test.gammaTable
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  
