# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:46:31 2018

明明都寫 Matrix 但都設 Array
"""
"""
class:
    https://www.brilliantcode.net/761/python-3-6-class/
"""
import numpy as np
#import warnings
#%% from 1008.3_4.0_gamma_RecognitionProblem.py edit
#class HidenMarkovModel_gamma():
class HMM_Dice_gamma():
    def __init__(self, initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput):
        
        # state
        self.initialStateProb   = initialStateProb
        # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
        self.stateChangeMatrix = stateChangeMatrix
        # [i][j] 在state i 時，生成j的機率    
        self.probabilityMatrix = probabilityMatrix
        
        # output
        self.stateName   = stateName
        self.stateOutput = stateOutput
#        # output
#        self.stateName = ['fair', 'unfair']
##        self.stateOutput = np.array([str(i) for i in range(1, len(self.probabilityMatrix[0,:])+1)])
#        self.stateOutput = np.array([ i for i in range(1, len(self.probabilityMatrix[0,:])+1)]) #這樣就可以用非字串格式了
        #
        self.stateNumber = len(self.initialStateProb) 
        #驗證資料正確性
        assert len(self.initialStateProb) == len(self.stateChangeMatrix)
        assert len(self.initialStateProb) == len(self.probabilityMatrix)
        assert len(self.initialStateProb) == len(self.stateName)
        assert len(self.initialStateProb) == len(self.stateChangeMatrix[:,0])
#        if (not len(self.initialStateProb) == len(self.stateChangeMatrix)) \
#            or (not len(self.initialStateProb) == len(self.probabilityMatrix)) \
#            or (not len(self.initialStateProb) == len(self.stateName)):
#            raise AssertionError("State 數量未對上")
#        if (not len(self.initialStateProb) == len(self.stateChangeMatrix[:,0])): 
#            raise AssertionError("輸出 數量未對上")
        return

#    def a_prob(self, preState, nextState):
#        """(未使用)從 preState 換到 nextState 的機率"""
#        prob = self.stateChangeMatrix[preState, nextState]
#        return prob
#    
#    def b_prob(self, state, output):
#        """(未使用)從 state 生出 output 的機率"""
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
    
    def PredictUseAlpha(self, target, boolPrint = True):
        """ 將 Alpha 最後兩組相加，便是所求字串發生機率"""
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
        #收尾 t = 0，生成第一個target的機率 #交給真的收尾輸出
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
        prob_PrO      = self.PredictUseAlpha(target, boolPrint = False)
        prob_PrO_beta = self.PredictUseBeta (target, boolPrint = False)
        alpha = self.alphaTable
        beta = self.betaTable
#        prob_PrO = beta[0,:].sum(dtype = np.float32) #alpha[-1,:].sum(dtype = np.float32)
        if not prob_PrO == prob_PrO_beta:
#            raise AssertionError("alpha(",prob_PrO,"),beta(",prob_PrO_beta,")結果不同")
#            warnings.warn("Warning...........Message")
#            warnings.warn("alpha( "+str(prob_PrO)," ),beta( "+str(prob_PrO_beta)+" )結果不同")
            print("Warnings:", "alpha(",prob_PrO,"),beta(",prob_PrO_beta,")結果不同")
        #gamma - time(-1) - state
        gamma = np.zeros((len(target), self.stateNumber), dtype = np.float32)
        gamma = (alpha * beta) / prob_PrO
        #另外轉存
        self.gammaTable = gamma
        self.prob_PrO = prob_PrO #new
        return gamma
    
    def Predict_optimalStateSequence_useGamma(self, target):
        """ """
        gamma = self.CalGammaTable(target)
        seqLis = []
        probOfOptimalSeq = 1
#        for t in range(len(target)):
#            indexTmp = gamma[t, :].argmax()
#            seqLis.append(self.stateName[indexTmp])
        for i, indexTmp in enumerate(gamma.argmax(axis = 1)):
            probOfOptimalSeq *= gamma[i][indexTmp]
            seqLis.append(self.stateName[indexTmp])
        print('"',target,'"',"'s Optimal State Sequence's Probability:", probOfOptimalSeq)
        print('"',target,'"',"'s Optimal State Sequence is", seqLis)
        return
#%%
class HMM_Dice_BaumWelch(HMM_Dice_gamma):
    def __init__(self, initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput):
        super().__init__(initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput)
        
        return
    def CalZetaTalbe(self, target):
        """ """
        gamma = self.CalGammaTable(target)
        
        zeta = np.zeros((len(target)-1, self.stateNumber, self.stateNumber))
        
        
        return
    def Train(self, trainData):
        for dataTmp in trainData[0]:
            outputState = dataTmp[0]
            target = dataTmp[1]
            
            self.CalZetaTalbe(target)
            
        return
#%%
def ChangeFormatToUse(inputArr):
    """ (多寫的) 將共用格式 狀態(array)、輸出(array) 換成  狀態(array)、輸出(string) """
#    print(inputArr.shape[1])
    outputLis = []
    for i in range(inputArr.shape[0]):
        outputLis.append([[] for i in range(inputArr.shape[1])])
        data = inputArr[i]
        outputLis[i][0] = data[0] #speciData[i]
        outputLis[i][1]= "".join(list(map(str, data[1])))
    return outputLis
#%%
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
    #%% 基本設定
    #state
    initialStateProb  = np.array([0.5, 0.5]) #[fair, unfair]
    # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
    stateChangeMatrix = np.array([[0.95, 0.05],
                                  [0.10, 0.90]]) 
    #[i][j] 在state i 時，生成j的機率    
    probabilityMatrix = np.array([[ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6], #fair 
                                  [1/10, 1/10, 1/10, 1/10, 1/10,  1/2]])#unfair
    # name
    stateName = ['fair', 'unfair']
    stateOutput = np.array([ str(i) for i in range(1, len(probabilityMatrix[0,:])+1)]) 
#    stateOutput = np.array([ i for i in range(1, len(probabilityMatrix[0,:])+1)]) #這樣就可以用非字串格式了
    #%% 輸入
    trainData = np.load("100Observation.npy")
    testData  = np.load("100seqTest.npy")
#    #%% 預處理- 改一行 class code 就可以免了~
#    trainData_edit = ChangeFormatToUse(trainData)
#    testData_edit  = ChangeFormatToUse(testData)
    #%%
    test = HMM_Dice_BaumWelch(initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput)
    test.CalZetaTalbe(target = "132456")
    alpha, beta, gamma = test.alphaTable, test.betaTable, test.gammaTable
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  
