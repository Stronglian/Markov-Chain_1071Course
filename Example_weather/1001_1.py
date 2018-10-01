# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 14:48:13 2018

0: state 1, 1:state 2, 2:state 3

"""

import numpy as np
import random

class HidenMarkovModel():
    def __init__(self):
        self.frequency = 10000
        self.randomNum = 10000
        
#        self.tableO = { '↑':0, '↓':1, '-':2}
        
        self.stateChangeMatrix = np.array([[0.6, 0.2, 0.2],
                                           [0.5, 0.3, 0.2],
                                           [0.4, 0.1, 0.5,]]) # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
        self.probabilityMatrix = np.array([[0.7, 0.1, 0.2],
                                           [0.1, 0.6, 0.3],
                                           [0.3, 0.3, 0.4]]) #[i][j] 在state i 時，生成j的機率
        self.initialStateProb   = np.array([0.5, 0.2, 0.3])
        return
    def NextState(self, currectState = None, probState = None):
        #依照機率算下一個狀態
        if currectState != None:#probState == None:# and #分辨 initial
            probState = self.stateChangeMatrix[currectState]
        
        tempRan = random.randrange(0, self.randomNum)
        if tempRan < self.randomNum*probState[0]:
            return 0
        elif tempRan < self.randomNum*(probState[0]+probState[1]):
            return 1
        else:
            return 2
    def OutputAtState(self, currectState):
        #依照機率與現有狀態得出現機率
        probO = self.probabilityMatrix[currectState]
        
        tempRan = random.randrange(0, self.randomNum)
        if tempRan < self.randomNum*probO[0]:
            return '↑'
        elif tempRan < self.randomNum*(probO[0]+probO[1]):
            return '↓'
        else:
            return '-'
        
    def CalOneRound(self, repeatTimes=5):
        #
        outputSeq = ''
        #初始
        currentState = self.NextState(probState = self.initialStateProb)
        outputSeq += self.OutputAtState(currentState)
        #重複
        for t in range(repeatTimes-1):
            currentState = self.NextState(currectState = currentState)
            outputSeq += self.OutputAtState(currentState)
        return outputSeq
    def Predict(self, targetSeq = '↑↑↑↑↑', printTF = True):
        #計算為目標序列(targetSeq)的機率
        repeatTimes = len(targetSeq)
        targetNum = 0
        for t in range(self.frequency):
            preO = self.CalOneRound(repeatTimes = repeatTimes)
            if preO == targetSeq:
                targetNum +=1
        if printTF:
            print('The', targetSeq,'appeared',targetNum,'times in',self.frequency,'times.')
    
        return (float(targetNum)/ self.frequency)
    def CalMaxMin(self, times = 100):
        #跑 times 次，找最大最小機率
        minP, maxP = 1, 0
        for i in range(times):
            num = test.Predict(printTF=False)
            if num > maxP:
                maxP = num
            if num < minP:
                minP = num
        print('run',times,'times, max:',maxP,', min:', minP, )

if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START")
    test = HidenMarkovModel()
    
    print(test.Predict())

    endTime = time.time()
    print('END,', 'It takes', endTime-startTime ,'sec.')