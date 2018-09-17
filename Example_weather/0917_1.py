# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:57:03 2018

@author: StrongPria
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
        self.days = 3
        self.firstDay = 'S'
    def NextWeather(self, today):
        assert today in self.tableW.keys()
        indexW = self.tableW[today]
        probW = self.probabilityMatrix[indexW]
        tempRan = random.randrange(0, self.randomNum)
        if tempRan < self.randomNum*probW[0]:
            return 'R'
        elif tempRan < self.randomNum*(probW[0]+probW[1]):
            return 'C'
        else:
            return 'S'
    def CalOneRound(self):
        weaherSeq = ''
        todaydayW = self.firstDay
        for d in range(self.days):
            todaydayW = self.NextWeather(todaydayW)
            weaherSeq += todaydayW
        return weaherSeq



if __name__ == '__main__' :
    print("START")
    test = MarkovChain_weather()
#    print (test.NextWeather('R'))
    print(test.CalOneRound())