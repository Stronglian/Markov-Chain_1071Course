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
import prettytable
from c1105_3_7_0_BaumWelchRe_BaumFormulas_extend import * 
#%%
def TrainAllAndPrint(trainData, testData, parmSet = None, field_names = ["pi", "A", "B"], boolTablePrint = True):
    if not parmSet is None:
        assert len(parmSet) == 3, "parmSet 參數內有 3: (initialStateProb, stateChangeMatrix, probabilityMatrix)"
        initialStateProb, stateChangeMatrix, probabilityMatrix = parmSet
    #建模
    HMM_B = HMM_Dice_BaumWelch(initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput)
    HMM_trainResult = HMM_B.Train(trainData)
    #結果
    result_test_rate  = HMM_B.Test(testData,  boolPrint=False)
    result_train_rate = HMM_B.Test(trainData, boolPrint=False)
    #建表
    ta = prettytable.PrettyTable()
    ta.field_names = field_names
    ta.add_row(HMM_trainResult)
    # 輸出
    print("Testing Rate Accuracy is", result_test_rate)
    print("Training Rate Accuracy is", result_train_rate)
    if boolTablePrint:
        print(ta)
    return
#%% 流程
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
#%% 基本設定 - 原本
    #state
    initialStateProb   = np.array([0.6, 0.4])#[fair, unfair]
    # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
    stateChangeMatrix = np.array([[0.95, 0.05],
                                  [0.10, 0.90]]) 
    #[i][j] 在state i 時，生成j的機率    
    probabilityMatrix = np.array([[ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6], #fair 
                                  [1/10, 1/10, 1/10, 1/10, 1/10,  1/2]])#unfair
#%% 基本設定 - 題目
##    # state
#    initialStateProb  = np.array([0.5, 0.5]) #[fair, unfair]
##    # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
#    stateChangeMatrix = np.array([[0.5, 0.5],
#                                  [0.5, 0.5]]) 
##    #[i][j] 在state i 時，生成j的機率    
#    probabilityMatrix = np.array([[ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6], #fair 
#                                  [ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6]])#unfair
#%% 基本設定 - 共同
    # name
    stateName = ['fair', 'unfair']
#    stateOutput = np.array([ str(i) for i in range(1, len(probabilityMatrix[0,:])+1)]) 
    stateOutput = np.array([ i for i in range(1, len(probabilityMatrix[0,:])+1)]) #這樣就可以用非字串格式了
#%% 輸入
#    trainData = np.load("100Observation.npy")
##    trainData = np.load("300SeqTrain.npy")
##    trainData = np.load("300_300seqTraining.npy")
#    testData  = np.load("100seqTest.npy")
#%% 開始 - 基本
#    HMM_B = HMM_Dice_BaumWelch(initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput)
##    # train
#    HMM_trainResult = HMM_B.Train(trainData)
##    # test
#    test_result = HMM_B.Test(testData)
##    # coe
###    HMM_alpha, HMM_beta, HMM_gamma = HMM_B.alphaTable, HMM_B.betaTable, HMM_B.gammaTable
###    HMM_zeta = HMM_B.zetaTable
    
#%% 全測試 - 2018/11/12 單一 - 參數換過了
#    trainDataName = "100Observation.npy"
#    testDataName  = "100seqTest.npy"
#    trainData = np.load(trainDataName)
#    testData  = np.load(testDataName)
#    print("TrainData:", trainDataName, "TestData:", testDataName)
##    print("Training Rate:")
#    TrainAllAndPrint(trainData, trainData, [initialStateProb, stateChangeMatrix, probabilityMatrix])
##    print("Testing Rate")
##    TrainAllAndPrint(trainData, testData,  [initialStateProb, stateChangeMatrix, probabilityMatrix])
#    print("\n\n", "="*50)
#%% 全測試 - 參數設置
    trainDataList = ["100Observation.npy", "300SeqTrain.npy", "300_300seqTraining.npy"]
    testDataList  = ["100seqTest.npy"]
#%% 全測試 - 2018/11/19 - 參數可能換過
#    for trainDataName in trainDataList:
#        print("train dataset:", trainDataName)
#        trainData = np.load(trainDataName)
#        for testDataName in testDataList:
#            print("test dataset:", testDataName)
#            testData  = np.load(testDataName)
#            TrainAllAndPrint(trainData, testData, [initialStateProb, stateChangeMatrix, probabilityMatrix])
#            print("\n")
#    print("\n\n", "="*50)
#%% 全測試 - 2018/11/26
    # 指定參數
    parmSetList = [  [np.array([0.6, 0.4]), 
                      np.array([[0.95, 0.05],  [0.1, 0.9]]), 
                      np.array([[0.18, 0.18, 0.16, 0.16, 0.16, 0.16],[0.18, 0.18, 0.16, 0.16, 0.16, 0.16]])],
                     [np.array([0.4, 0.6]), 
                      np.array([[0.95, 0.05],  [0.1, 0.9]]), 
                      np.array([[0.168, 0.168, 0.166, 0.166, 0.166, 0.166],[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])],
                     [np.array([0.6, 0.4]), 
                      np.array([[0.95, 0.05],  [0.1, 0.9]]), 
                      np.array([[0.168, 0.168, 0.166, 0.166, 0.166, 0.166],[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])]                    
                ]
    # 指定資料
    trainDataName = "300_300seqTraining.npy"
    testDataName  = "300SeqTrain.npy"
    trainData = np.load(trainDataName)
    testData  = np.load(testDataName)
    print("TrainData:", trainDataName, "TestData:", testDataName)
    print("Training Rate:","(use",trainDataName,")")
    print("Testing Rate:","(use",testDataName,")")
    # RUN
    for i, parmSet in enumerate(parmSetList):
        print("\n", i+1, sep="")
#        TrainAllAndPrint(trainData.copy(), trainData.copy(), parmSet)
        TrainAllAndPrint(trainData.copy(), testData.copy(), parmSet)
        
#%% 收尾
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  
    

