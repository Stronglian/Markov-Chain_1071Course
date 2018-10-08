# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:57:03 2018
"""
"""
assert:
    https://blog.csdn.net/humanking7/article/details/45950781
random doc:
    https://docs.python.org/2/library/random.html
"""
import numpy as np
import random

class MarkovChain_weather():
    def __init__(self):
        #Rainy(R), Cloudy(C), Sunny(S)
        self.tableW = { 'R':0, 'C':1, 'S':2}
        self.probabilityMatrix = np.array([[0.4, 0.3, 0.3],
                                           [0.2, 0.6, 0.2],
                                           [0.1, 0.1, 0.8]])
        self.frequency = 10000
        self.randomNum = 1000
        
        self.firstDayW = 'S'
#        self.days = 3
        return
    def NextWeather(self, todayW):
        #依照機率表(probabilityMatrix)與當天天氣(todayW)與猜測隔天天氣。
        assert todayW in self.tableW.keys()
        indexW = self.tableW[todayW]
        probW = self.probabilityMatrix[indexW]
        
        tempRan = random.randrange(0, self.randomNum)
        if tempRan < self.randomNum*probW[0]:
            return 'R'
        elif tempRan < self.randomNum*(probW[0]+probW[1]):
            return 'C'
        else:
            return 'S'
    def CalOneRound(self, days=3):#, firstDayW='S'):
        #計算單次連續天數(days)的天氣可能為何(weaherSeq)。
        weaherSeq = ''
        todaydayW = self.firstDayW
#        todaydayW = firstDayW
        for d in range(days):
            todaydayW = self.NextWeather(todaydayW)
            weaherSeq += todaydayW
        return weaherSeq
    def Predict(self, firstDayW='S', targetSeq = 'SSSRRSCS'):
        #計算指定天數(len(targetSeq))下，當今天天氣為 firstDayW，往後天氣會依序發生(targetSeq)的機率。
        assert firstDayW in self.tableW.keys()
        self.firstDayW = firstDayW
        days = len(targetSeq)
        targetNum = 0
        for d in range(self.frequency):
            preW = self.CalOneRound(days = days)
            if preW == targetSeq:
                targetNum +=1
       # print('The', targetSeq,'appeared',targetNum,'times in',self.frequency,'times.')
    
        return round(targetNum*100.0 / self.frequency, 3)

if __name__ == '__main__' :
    import time
    startTime = time.time()
    print("START")
    test = MarkovChain_weather()
#   
#    print (test.NextWeather('R'))
    print(test.Predict(firstDayW='S', targetSeq = 'SRC'),'%')
    
    endTime = time.time()
    print('END,', 'It takes', endTime-startTime ,'sec.')