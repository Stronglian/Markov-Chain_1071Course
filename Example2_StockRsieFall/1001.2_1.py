# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 14:48:13 2018

0: state 1, 1:state 2, 2:state 3

嘗試做第二題 at slide P.22
"""
"""
先做了兩項相似函數的整合，但仍保留原有的名稱
作業二 ，達成 targetSeq 的最佳路徑(state)為何?(未進行)
紀錄array：路徑、機率、結果
找符合結果的最大機率，輸出路徑
"""

import numpy as np


class HidenMarkovModel():
    def __init__(self):
        self.frequency = 10000
        self.randomNum = 10000
        # 0, 1, 2 # [i][j] 在state i 時，換到 state j 的機率
        self.stateChangeMatrix = np.array([[0.6, 0.2, 0.2],
                                           [0.5, 0.3, 0.2],
                                           [0.4, 0.1, 0.5]])
        # [i][j] 在 state i 時，生成 j 的機率
        self.probabilityMatrix = np.array([[0.7, 0.1, 0.2],
                                           [0.1, 0.6, 0.3],
                                           [0.3, 0.3, 0.4]])
        self.initialStateProb = np.array([0.5, 0.2, 0.3])
        return

    def CalOutput(self, currectState: np.int, probM: np.ndarray, output: list):
        """ 算機率的共用函數"""
        if not len(probM) == len(output):
            print(currectState, probM, output)
            raise AssertionError

        tempRan = np.random.randint(0, self.randomNum)
        for i in range(len(output)):
            if tempRan < self.randomNum * probM[:i + 1].sum():  #總和到i項
                return output[i]
                break  #多的

    def NextState(self,
                  currectState: np.int = None,
                  probState: np.ndarray = None):
        """
        依照機率算下一個狀態。
        有 probState，currectState 為空，為初始狀態。
        兩者皆有為初始狀態。
        """
        #分辨 initial
        if currectState != None:  #probState == None:# and
            probState = self.stateChangeMatrix[currectState]

        return self.CalOutput(currectState=currectState,
                              probM=probState,
                              output=[0, 1, 2])

    def OutputAtState(self, currectState: np.int):
        """依照機率與現有狀態得出現機率"""
        return self.CalOutput(currectState=currectState,
                              probM=self.probabilityMatrix[currectState],
                              output=["↑", "↓", "-"])

    def CalOneRound(self, repeatTimes: np.int = 5):
        """"""
        outputSeq = ""
        #初始
        currentState = self.NextState(probState=self.initialStateProb)
        outputSeq += self.OutputAtState(currentState)
        #重複
        for t in range(repeatTimes - 1):  #-1因為第一次已經跑了
            currentState = self.NextState(currectState=currentState)
            outputSeq += self.OutputAtState(currentState)
        return outputSeq

    def Predict(self, targetSeq: str, printTF: bool = True):
        """計算為目標序列(targetSeq)的機率"""
        repeatTimes = len(targetSeq)
        targetNum = 0
        for t in range(self.frequency):
            preO = self.CalOneRound(repeatTimes=repeatTimes)
            if preO == targetSeq:
                targetNum += 1
        if printTF:
            print("The", targetSeq, "appeared", targetNum, "times in",
                  self.frequency, "times.")

        return (float(targetNum) / self.frequency)

    def CalMaxMin(self, targetSeq, times=10, printTF=False):
        """ 跑 times 次，找最大最小機率 """
        minP, maxP, avgP = 1, 0, 0
        for i in range(times):
            num = test.Predict(targetSeq=targetSeq, printTF=printTF)
            avgP += num
            if num > maxP:
                maxP = num
            if num < minP:
                minP = num
        avgP /= times
        print(f"run {times} times, max: {maxP}, min: {minP} avg: {avgP}")
        return


if __name__ == "__main__":
    import time
    startTime = time.time()
    print("START")
    test = HidenMarkovModel()

    #    print("Probability is", test.Predict(targetSeq="↑↑↑↑↑"))
    test.CalMaxMin(targetSeq="↑↑↑↑↑", printTF=True)

    endTime = time.time()
    print("\n\n\nEND, It takes {endTime - startTime} sec.")
