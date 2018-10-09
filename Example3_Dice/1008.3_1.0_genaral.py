# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 18:38:23 2018

@title: unfair dice
"""


import numpy as np
import random

class HidenMarkovModel_genaral():
    def __init__(self):
        self.frequency = 100000
        self.randomNum = 10000
        # 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
        self.stateChangeMatrix = np.array([[0.95, 0.05],
                                           [0.10, 0.90]]) 
        #[i][j] 在state i 時，生成j的機率    
        self.probabilityMatrix = np.array([[ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6], #fair 
                                           [1/10, 1/10, 1/10, 1/10, 1/10,  1/2]])#unfair 
        self.initialStateProb   = np.array([0.6, 0.4]) #[fair, unfair]
        self.stateOutput = [str(i) for i in range(1, 6+1)]
        return
    def CalOutput(self, currectState = None, probM = None, output=[]):
        """ 算機率的共用函數"""
#        assert len(probM) == len(output) #確認輸入正確
        if not len(probM) == len(output):
            print(currectState, probM, output)
            raise AssertionError
        tempRan = random.randrange(0, self.randomNum)
        for i in range(len(output)):
            if tempRan < self.randomNum*probM[:i+1].sum(): #總和到i項
                return output[i]
                break #多的
    def NextState(self, currectState = None, probState = None):
        """依照機率算下一個狀態(使用的骰子)"""
        #分辨 initial
        if currectState != None:#probState == None:# and 
            probState = self.stateChangeMatrix[currectState]
        
        return self.CalOutput(currectState = currectState, 
                         probM = probState, 
                         output=[i for i in range(len(probState))]) #可共用
    def OutputAtState(self, currectState):
        """依照機率與現有狀態得出現機率"""
        return self.CalOutput(currectState = currectState, 
                         probM = self.probabilityMatrix[currectState], 
                         output= self.stateOutput)
    def CalOneRound(self, repeatTimes=5):
        """執行一次的內容"""
        outputSeq = ''
        #初始
        currentState = self.NextState(probState = self.initialStateProb)
        outputSeq += self.OutputAtState(currentState)
        #重複
        for t in range(repeatTimes-1):#-1因為第一次已經跑了
            currentState = self.NextState(currectState = currentState)
            outputSeq += self.OutputAtState(currentState)
        return outputSeq   
    def Predict(self, targetSeq, printTF = True):
        """計算為目標序列(targetSeq)的機率"""
        repeatTimes = len(targetSeq)
        targetNum = 0
        for t in range(self.frequency):
            print('\b'*6, round(t*100.0/self.frequency,2),'%', flush = True, end='')
            preO = self.CalOneRound(repeatTimes = repeatTimes)
            if preO == targetSeq:
                targetNum +=1
        if printTF:
            print('The', targetSeq,'appeared',targetNum,'times in',self.frequency,'times.')
    
        return (float(targetNum)/ self.frequency)
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START")
    test = HidenMarkovModel_genaral()
    
    print("Probability is", test.Predict(targetSeq = '123456'))
    

    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')