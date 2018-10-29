# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:54:29 2018
生成 100 組，長度100的字串
驗證其optimal
"""

import numpy as np

from c1029_3_1_1_genaralForAccuracyCal import HidenMarkovModel_genaral
from c1015_3_5_0_ViterbiAlgorithm import HidenMarkovModel_Viterbi

#np.random.seed(3)

class NumberRecord_MaxMinCount():
    def __init__(self):#, maxVal = 100.0, minVal = 0.0):
        """ 定義會遇到的最大與最小，直接拿第一次做標準"""
#        self.maxNum = minVal
#        self.minNum = maxVal
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
        """ """
        #記錄到 dict
        if round(inputNum, roundAxis) not in self.DictCount.keys():
            self.DictCount[round(inputNum, roundAxis)] = 1
        else:
            self.DictCount[round(inputNum, roundAxis)] += 1
        return
    
    def ShowResult(self, printSting="", lineNum = 0.5):
        print(printSting, "max:", self.maxNum, " min:", self.minNum, "avg:", self.sumTmp/self.countAmount)
        #計數
        countNum = 0
        for i_key in self.DictCount.keys():
            if i_key < 0.5:
                countNum += self.DictCount[i_key]
#        print(list(self.DictCount.keys()))
        print(printSting, "<", lineNum, "have", countNum)
        return
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START\n\n")
    
    genaral = HidenMarkovModel_genaral()
    viterbi = HidenMarkovModel_Viterbi()
    recordC = NumberRecord_MaxMinCount()
    
    stringAmount = 100 #100組
    stringLen    = 100 #長度100
    
    accuracySum = 0
    for i in range(stringAmount):
        # 生成
        outputString, outputState = genaral.CalOneRound(stringLen)
        stringProb, optimalStateSeq = viterbi.Predict_optimalStateSequence_useRoPsi(outputString, boolPrint=False)
        # 計算
    #    print(sum(abs(optimalStateSeq - outputState))) #錯誤個數
#        accuracyV = (stringLen - sum(abs(optimalStateSeq - outputState)))/stringLen
        accuracyV = (list(optimalStateSeq - outputState).count(0))/float(stringLen) #以正確來算
        # 紀錄
        recordC.RecordFunc(accuracyV)
#        accuracySum += accuracyV #全部讓class處理
        
#    print(accuracySum/stringAmount)
    recordC.ShowResult(printSting = "Accuracy")
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')  