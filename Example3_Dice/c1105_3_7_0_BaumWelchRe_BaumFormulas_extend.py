# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:46:31 2018

明明都變數名稱寫 Matrix ，但都是 Array
"""
"""
class:
    https://www.brilliantcode.net/761/python-3-6-class/
wiki:
    https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
someone's slide:
    https://people.cs.umass.edu/~mccallum/courses/inlp2004a/lect10-hmm2.pdf
"""
import numpy as np
#import warnings
#%% from 1008.3_4.0_gamma_RecognitionProblem.py edit
#class HidenMarkovModel_gamma():
class HMM_Dice_gamma():
    def __init__(self, initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput):
        # state
        self.initialStateProb   = initialStateProb.copy()
        # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
        self.stateChangeMatrix = stateChangeMatrix.copy()
        # [i][j] 在state i 時，生成j的機率    
        self.probabilityMatrix = probabilityMatrix.copy()
        
        # output
        self.stateName    = stateName.copy()
        self.stateOutput  = stateOutput.copy()
        #
        self.stateNumber  = len(self.initialStateProb) 
        self.outputNumber = len(probabilityMatrix[0,:])
        #驗證資料正確性
        assert len(self.initialStateProb) == len(self.stateChangeMatrix),      "State 數量未對上"
        assert len(self.initialStateProb) == len(self.probabilityMatrix),      "State 數量未對上"
        assert len(self.initialStateProb) == len(self.stateName),              "State 數量未對上"
        assert len(self.initialStateProb) == len(self.stateChangeMatrix[:,0]), "輸出 數量未對上"
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
        alpha = np.zeros((len(target), self.stateNumber), dtype = np.float64)
        #初始 t = 0，生成第一個target的機率
        alpha[0, :] = self.initialStateProb * self.probabilityMatrix[:, self.stateOutput == target[0]].T
        #剩下的字串
        for t in range(1, len(target)):
            tempSum = np.multiply(alpha[t-1,:], self.stateChangeMatrix.T).sum(axis = 1, dtype = np.float64)
            #print(tempSum)
            alpha[t, :] = np.multiply(tempSum, self.probabilityMatrix[:, self.stateOutput == target[t]].T)#self.b_prob(j, target[t])
        #另外轉存
        self.alphaTable = alpha
        return alpha
    
    def PredictUseAlpha(self, target, boolPrint = True):
        """ 將 Alpha 最後兩組相加，便是所求字串發生機率"""
        alpha = self.CalAlphaTable(target)
        prob_ProO_alpha = alpha[-1,:].sum(dtype = np.float64)
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
        beta = np.zeros((len(target), self.stateNumber), dtype = np.float64)
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
        #另外轉存
        self.betaTable = beta
        return beta
    
    def PredictUseBeta(self, target, boolPrint=True):
        """ 將 Beta 最前兩組相加，便是所求"""
        beta = self.CalBetaTable(target)
        prob_PrO_bata = (beta[0,:] * self.initialStateProb * self.probabilityMatrix[:, self.stateOutput == target[0]].T).sum(dtype = np.float64)
        if boolPrint:
            print('betaTable:\n',beta)
            print('"',target,'"',"'s Probability:", prob_PrO_bata)
        return prob_PrO_bata
    
    def CalGammaTable(self, target, boolWarningShow = False):
        """ 利用 alpha、beta 來算該 state 發生機率，進而推導最佳 State 順序"""
#        prob_PrO = beta[0,:].sum(dtype = np.float64) #alpha[-1,:].sum(dtype = np.float64)
        prob_PrO_alpha = self.PredictUseAlpha(target, boolPrint = False)
        prob_PrO_beta  = self.PredictUseBeta (target, boolPrint = False)
        alpha = self.alphaTable
        beta  = self.betaTable
        if (not prob_PrO_alpha == prob_PrO_beta) and boolWarningShow:
            print("Warnings:", target, "'s alpha(",prob_PrO_alpha,"),beta(",prob_PrO_beta,")結果不同")
#        print("==>:", target, "'s alpha(",prob_PrO,"),beta(",prob_PrO_beta,")")
        #gamma - time(-1) - state
        gamma = np.zeros((len(target), self.stateNumber), dtype = np.float64)
        gamma = (alpha * beta) / prob_PrO_alpha
        #另外轉存
        self.gammaTable = gamma
        self.prob_PrO = prob_PrO_alpha #new
        return gamma
    
    def Predict_optimalStateSequence_useGamma(self, target):
        """ """
        gamma = self.CalGammaTable(target)
        seqLis = []
        probOfOptimalSeq = 1
        for i, indexTmp in enumerate(gamma.argmax(axis = 1)):
            probOfOptimalSeq *= gamma[i][indexTmp]
            seqLis.append(self.stateName[indexTmp])
        print('"',target,'"',"'s Optimal State Sequence's Probability:", probOfOptimalSeq)
        print('"',target,'"',"'s Optimal State Sequence is", seqLis)
        return probOfOptimalSeq, seqLis

#%% from c1015_3_5_1_ViterbiAlgorithm_UPDATE.py edit
#class HidenMarkovModel_Viterbi():
class HMM_Dice_Viterbi(HMM_Dice_gamma):
    def __init__(self, initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput):
        super().__init__(initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput)
        return
    
    def CalRoPsiTable(self, target):
        """ """
        lenTarget = len(target)
        # step 0 - table #table - time(-1) - state #ρ (ro)、ψ(psi)
        ro  = np.zeros((len(target), self.stateNumber))
        psi = np.ones((len(target), self.stateNumber)) * -1
        # step 1 – Initialization #psi DONE before
        ro[0,:]  = self.initialStateProb.T * self.probabilityMatrix[:, self.stateOutput == target[0]].T[0]
        
        # step 2 – Recursion
        tmpArr = np.zeros((self.stateNumber, self.stateNumber))
        for t in range(1, len(target)):
            tmpArr = np.zeros_like(tmpArr)
            ### 法二
            tmpArr[:, :] = ro[t-1, :] * self.stateChangeMatrix[:, :].T * self.probabilityMatrix[:, self.stateOutput == target[t]]
#            print(t, "\n", tmpArr)
            ro[t, :]  = tmpArr.max(axis = 1)
            psi[t, :] = tmpArr.argmax(axis = 1) 
        #另外轉存
        self.roTable  = ro.copy()
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
            stateSeqIndex[t-1] = psi[t, stateSeqIndex[t]]
#        print(stateSeqIndex)
        return ro, psi, P_star_all, stateSeqIndex
    
    def Predict_optimalStateSequence_useViterbi(self,target, boolPrint = True):
        """ """
        ro, psi, bestProb, bestStateSquenceIndex = self.CalRoPsiTable(target)
        #The best state sequence having the highest probability
        if boolPrint:
            print('"',target,'"',"'s Probability:", bestProb) 
        seqLis = []
        for indexTmp in bestStateSquenceIndex: 
            seqLis.append(self.stateName[indexTmp])
        if boolPrint:
            print('"',target,'"',"'s Optimal State Sequence is", seqLis)
        return bestProb, bestStateSquenceIndex
#%%
class HMM_Dice_BaumWelch(HMM_Dice_Viterbi):
    def __init__(self, initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput):
        super().__init__(initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput)
        return
    
    def CalZetaTalbe(self, target):
        """ 路徑發生機率"""
        gamma    = self.CalGammaTable(target)
        alpha    = self.alphaTable
        beta     = self.betaTable
        prob_PrO = self.prob_PrO
        #zeta - time(t)(-1) - currectState - nextState
        zeta = np.zeros((len(target)-1, self.stateNumber, self.stateNumber), dtype = np.float64)
        # 法一 - for
        for t in range(len(target) - 1):
            for i in range(self.stateNumber):
                for j in range(self.stateNumber):
                    zeta[t, i, j] = alpha[t, i] * self.stateChangeMatrix[i, j] * \
                    self.probabilityMatrix[j, self.stateOutput == target[t+1]] * beta[t+1, j] / prob_PrO
        self.zetaTable = zeta.copy()
        return zeta
    
    def Train(self, trainData):
        """ 利用訓練資料集輸入 zeta 來重新訓練各參數"""
        # 代稱，so NO copy()
        Pi = self.initialStateProb
        A  = self.stateChangeMatrix
        B  = self.probabilityMatrix
        for dataTmp in trainData:
            # initial
#            outputState = dataTmp[0]
            inputTarget = dataTmp[1]
#            print(inputTarget, outputState, sep='\n')
            # E-STEP:evaluate alpha, beta, gamma, zeta
            zeta  = self.CalZetaTalbe(inputTarget)
            gamma = self.gammaTable
            # M-STEP:
            for i in range(self.stateNumber): 
                Pi[i] = self.gammaTable[0, i]
                for count_k in range(self.outputNumber): #計算輸出生成機率
                    B[i, count_k] = gamma[inputTarget == (count_k+1), i].sum()/gamma[:, i].sum() 
                for j in range(self.stateNumber): #計算狀態轉換率
                    A[i, j] = zeta[:-1, i, j].sum() /gamma[:-1, i].sum() 
                    
        return self.initialStateProb, self.stateChangeMatrix, self.probabilityMatrix
    
    def Test(self, testData, parmSet = None, boolPrint = True):
        """ 利用測試資料集，配合已經訓練完的參數，進行預測"""
        # 使用額外的參數
        tmpClass = self
        if not parmSet is None:
            print("額外參數")
            assert len(parmSet) == 3, "parmSet 參數內有 3: (initialStateProb, stateChangeMatrix, probabilityMatrix)"
            initialStateProb, stateChangeMatrix, probabilityMatrix = parmSet
            tmpClass = HMM_Dice_Viterbi(initialStateProb, stateChangeMatrix, probabilityMatrix, self.stateName, self.stateOutput)
        # 計算正確率
        finalProb = 0.0
        for outputState, inputTarget in testData:
            # take the coefficient after training to Viterbi
            optimal_Prob, optimal_Seq = tmpClass.Predict_optimalStateSequence_useViterbi(inputTarget, boolPrint=False)
            # Calculate the correct value for prediction
#            tmp = (outputState - optimal_Seq)
#            sameStateNumber = len( tmp[tmp == 0])
            sameStateNumber = np.count_nonzero((outputState - optimal_Seq)==0)
            finalProb += sameStateNumber / len(outputState)
        finalProb /= testData.shape[0]
        if boolPrint:
            print("Accuracy is", finalProb)
        return finalProb
#%% 預先處理用
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
#%% 流程
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
##%% 基本設定 - 原本
#    #state
#    initialStateProb   = np.array([0.6, 0.4])#[fair, unfair]
#    # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
#    stateChangeMatrix = np.array([[0.95, 0.05],
#                                  [0.10, 0.90]]) 
#    #[i][j] 在state i 時，生成j的機率    
#    probabilityMatrix = np.array([[ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6], #fair 
#                                  [1/10, 1/10, 1/10, 1/10, 1/10,  1/2]])#unfair
#%% 基本設定 - 題目
    # state
    initialStateProb  = np.array([0.5, 0.5]) #[fair, unfair]
    # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
    stateChangeMatrix = np.array([[0.5, 0.5],
                                  [0.5, 0.5]]) 
    #[i][j] 在state i 時，生成j的機率    
    probabilityMatrix = np.array([[ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6], #fair 
                                  [ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6]])#unfair
    # name
    stateName = ['fair', 'unfair']
#    stateOutput = np.array([ str(i) for i in range(1, len(probabilityMatrix[0,:])+1)]) 
    stateOutput = np.array([ i for i in range(1, len(probabilityMatrix[0,:])+1)]) #這樣就可以用非字串格式了
#%% 輸入
    trainData = np.load("100Observation.npy")
#    trainData = np.load("300SeqTrain.npy")
#    trainData = np.load("300_300seqTraining.npy")
    testData  = np.load("100seqTest.npy")
#    #%% 預處理- 改一行 class code 就可以免了~
#    trainData_edit = ChangeFormatToUse(trainData)
#    testData_edit  = ChangeFormatToUse(testData)
#%% 開始
    HMM_B = HMM_Dice_BaumWelch(initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput)
    # train
    HMM_trainResult = HMM_B.Train(trainData)
    # test
    print("testing rate:")
    _test_result = HMM_B.Test(testData)
    # train
    print("training rate:")
    _train_result = HMM_B.Test(trainData)
    # coe
#    HMM_alpha, HMM_beta, HMM_gamma = HMM_B.alphaTable, HMM_B.betaTable, HMM_B.gammaTable
#    HMM_zeta = HMM_B.zetaTable

#%% 收尾
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  
    

