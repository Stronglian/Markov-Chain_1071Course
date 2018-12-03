# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:43:12 2018

@author: StrongPria
"""

import numpy as np
from c1105_3_7_0_BaumWelchRe_BaumFormulas_extend import *
import os

#%% 檔案名稱
dataFolder = "datat98"
yangDataFileName = "yang98Data.npy" #旨陽 資料
changDataFileName = "chang98Data.npy" #思澄 資料
#%% 讀取答案
if yangDataFileName in os.listdir("./"):
    dataSet_yang = np.load(yangDataFileName).item()
elif dataFolder in os.listdir("./"):
    dataSet_yang = {}
    dataSet_yang["O"]       = np.load(dataFolder + "/" + "O_98.npy")        # observation sequence 
    dataSet_yang["S"]       = np.load(dataFolder + "/" + "S_98.npy")        # state sequence
    dataSet_yang["psi"]     = np.load(dataFolder + "/" + "phi_98.npy")      # Ψ
    dataSet_yang["ro"]     = np.load(dataFolder + "/" + "rho_98.npy")       # ρ 
    dataSet_yang["outputS"] = np.load(dataFolder + "/" + "outputS_98.npy") #預測出來的 state sequence S'
    np.save(yangDataFileName, dataSet_yang)
else:
    raise FileNotFoundError("無資料集", dataFolder, yangDataFileName)

#%% 參數
initialStateProb   = np.array([0.6, 0.4])#[fair, unfair]
# 0, 1, 2 #[i][j] 在state i 時，換到state j的機率
stateChangeMatrix = np.array([[0.95, 0.05],
                              [0.10, 0.90]]) 
#[i][j] 在state i 時，生成j的機率    
probabilityMatrix = np.array([[ 1/6,  1/6,  1/6,  1/6,  1/6,  1/6], #fair 
                              [1/10, 1/10, 1/10, 1/10, 1/10,  1/2]])#unfair

stateName = ['fair', 'unfair']
stateOutput = np.array([ i for i in range(1, len(probabilityMatrix[0,:])+1)])
#%% 使用
boolReTrain = True
if boolReTrain or (changDataFileName not in os.listdir("./")):
    dataSet_chang = {}
    viterbi = HMM_Dice_Viterbi(initialStateProb, stateChangeMatrix, probabilityMatrix, stateName, stateOutput)
    bestProb, bestStateSquenceIndex = viterbi.Predict_optimalStateSequence_useViterbi(dataSet_yang["O"], boolPrint = False)
    
    dataSet_chang["outputS"] = bestStateSquenceIndex #預測出來的 state sequence S
    dataSet_chang["ro"]  = viterbi.roTable  # Ψ
    dataSet_chang["psi"] = viterbi.psiTable # ρ
    
    np.save(changDataFileName, dataSet_chang)
else:
    dataSet_chang = np.load(changDataFileName).item()

#%% 比較
diffNum_outputS = len(np.nonzero(dataSet_chang["outputS"] - dataSet_yang["outputS"])[0])
diffNum_ro      = len(np.nonzero(dataSet_chang["ro"] - dataSet_yang["ro"])[0])
diffNum_psi     = len(np.nonzero(dataSet_chang["psi"] - dataSet_yang["psi"])[0])
print("Diff number is")
print("output:",diffNum_outputS)
print("ro", diffNum_ro)
print("psi:", diffNum_psi, "->", np.nonzero(dataSet_chang["psi"] - dataSet_yang["psi"]), "差異為初始值不同") #差異為初始值不同
print("與真實比較 - yang", len(np.nonzero(dataSet_yang["outputS"] - dataSet_yang["S"])[0]))
print("與真實比較 - chang", len(np.nonzero(dataSet_chang["outputS"] - dataSet_yang["S"])[0]))