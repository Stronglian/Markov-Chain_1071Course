# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:54:29 2018
生成 100 組，長度100的字串
驗證其optimal
"""

import numpy as np

from c1029_3_1_1_genaralForAccuracyCal import HidenMarkovModel_genaral
from c1015_3_5_1_ViterbiAlgorithm_UPDATE import HidenMarkovModel_Viterbi
#from c1015_3_5_0_ViterbiAlgorithm import HidenMarkovModel_Viterbi

#np.random.seed(3)

class NumberRecord_MaxMinCount():
    def __init__(self):#, maxVal = 100.0, minVal = 0.0):
        """ 定義會遇到的最大與最小，直接拿第一次做標準"""
        self.__boolFirst__ = True
        
        self.countAmount = 0
        self.sumTmp = 0.0
        self.DictCount = {}
        return
    
    def __SetFirstTime__(self, inputNum):
        """ 直接拿第一次做標準 """
        if self.__boolFirst__:
            self.__boolFirst__ = False
            self.maxNum = inputNum
            self.minNum = inputNum
#            return True
#        return False
        return
    
    def RecordFunc(self, inputNum):
        """ 數字進來做紀錄，主要功能：中途 flow、比大小 """
        #計算數量
        self.CountAmount(inputNum)
        self.countAmount += 1
        self.sumTmp += inputNum
        #第一次運行
        if self.__boolFirst__:
            self.__SetFirstTime__(inputNum)
            #因為第一次已經定義不用再比大小
            return
        #比大小
        if inputNum > self.maxNum:
            self.maxNum = inputNum
        elif inputNum < self.minNum:
            self.minNum = inputNum
        return
    
    def CountAmount(self, inputNum, roundAxis = 2):
        """ 記錄到 dict """
        if round(inputNum, roundAxis) not in self.DictCount.keys():
            self.DictCount[round(inputNum, roundAxis)] = 1
        else:
            self.DictCount[round(inputNum, roundAxis)] += 1
        return
    
    def ShowResult(self, printSting="", lineNum = 0.5):
        """ """
        print(printSting, "max:", self.maxNum, " min:", self.minNum, "avg:", round(self.sumTmp/self.countAmount,4), "there are", self.countAmount, "data.")
        #計數
        countNum = 0
        for i_key in self.DictCount.keys():
            if i_key < lineNum:
                countNum += self.DictCount[i_key]
        
        print(printSting, "<", lineNum, "have", countNum)
        return
    
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
    
    stringAmount = 100 #100組
    stringLen    = 100 #長度100
    
#    genaral = HidenMarkovModel_genaral()
    speciData = np.load("100Observation.npy")
    viterbi = HidenMarkovModel_Viterbi()
    recordC = NumberRecord_MaxMinCount()
    
    
    accuracySum = 0
#    for i in range(stringAmount):
    for i in range(1): #0.82
#        # 生成
#        outputString, outputState = genaral.CalOneRound(stringLen)
        #取資料
        data = speciData[i]
        outputState = data[0] #speciData[i]
        outputString = "".join(list(map(str, data[1])))
        # 預測
        stringProb, optimalStateSeq = viterbi.Predict_optimalStateSequence_useRoPsi(outputString, boolPrint=False)
        # 計算
        accuracyV = (list(optimalStateSeq - outputState).count(0))/float(stringLen) #以正確來算
        # 紀錄
        recordC.RecordFunc(accuracyV)
        
    recordC.ShowResult(printSting = "Accuracy")
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  