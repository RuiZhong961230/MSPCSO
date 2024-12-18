# import packages
import os
from enoppy.paper_based.pdo_2022 import *
import numpy as np
from copy import deepcopy

# Population size in Particle Swarm Optimization
PopSize = 100
DimSize = 100
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 20
MaxFEs = 1000 * DimSize

Pop = np.zeros((PopSize, DimSize))
Velocity = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
curFEs = 0
FuncNum = 1

His = np.zeros((PopSize, DimSize))
FitHis = np.zeros(PopSize)
H = 5
phi1 = [0.15] * H
phi2 = [0.15] * H
phi3 = [0.15] * H

# initialize the M randomly
def Initialization(func):
    global Pop, Velocity, FitPop, His, FitHis, phi1, phi2, phi3, H
    Velocity = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])
    His = deepcopy(Pop)
    FitHis = deepcopy(FitPop)
    phi1 = [0.15] * H
    phi2 = [0.15] * H
    phi3 = [0.15] * H


def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi

def MSPCSO(func):
    global Pop, Velocity, FitPop, His, FitHis, curFEs, phi1, phi2, phi3
    sequence = list(range(PopSize))
    np.random.shuffle(sequence)
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    Xmean = np.mean(Pop, axis=0)
    S_phi1, S_phi2, S_phi3 = [], [], []
    delta_phi1, delta_phi2, delta_phi3 = [], [], []
    idx_phi1, idx_phi2, idx_phi3 = np.random.randint(0, H, 3)
    for i in range(int(PopSize / 2)):
        idx1 = sequence[2 * i]
        idx2 = sequence[2 * i + 1]
        if FitPop[idx1] < FitPop[idx2]:
            Off[idx1] = deepcopy(Pop[idx1])
            FitOff[idx1] = FitPop[idx1]

            Vec1 = np.random.rand(DimSize) * (Pop[np.argmin(FitPop)] - Pop[idx2])
            Vec2 = np.random.rand(DimSize) * (Xmean - Pop[idx2])
            Vec3 = np.random.rand(DimSize) * (His[idx2] - Pop[idx2])

            r = np.random.rand()
            if r < 1 / 3:
                Phi1 = np.clip(np.random.normal(phi1[idx_phi1], 0.1), 0.001, 0.5)
                Velocity[idx2] = np.random.rand(DimSize) * Velocity[idx2] + np.random.rand(DimSize) * (Pop[idx1] - Pop[idx2]) + Phi1 * Vec1
                Off[idx2] = Pop[idx2] + Velocity[idx2]
                Off[idx2] = Check(Off[idx2])
                FitOff[idx2] = func(Off[idx2])
                curFEs += 1
                if FitOff[idx2] < FitHis[idx2]:  # Individual update
                    FitHis[idx2] = FitOff[idx2]
                    His[idx2] = deepcopy(Off[idx2])
                if FitOff[idx2] < FitPop[idx2]:  # Record success history
                    S_phi1.append(Phi1)
                    delta_phi1.append(FitPop[idx2] - FitOff[idx2])
            elif r < 2 / 3:
                Phi2 = np.clip(np.random.normal(phi2[idx_phi2], 0.1), 0.001, 0.5)
                Velocity[idx2] = np.random.rand(DimSize) * Velocity[idx2] + np.random.rand(DimSize) * (Pop[idx1] - Pop[idx2]) + Phi2 * Vec2
                Off[idx2] = Pop[idx2] + Velocity[idx2]
                Off[idx2] = Check(Off[idx2])
                FitOff[idx2] = func(Off[idx2])
                curFEs += 1
                if FitOff[idx2] < FitHis[idx2]:  # Individual update
                    FitHis[idx2] = FitOff[idx2]
                    His[idx2] = deepcopy(Off[idx2])
                if FitOff[idx2] < FitPop[idx2]:  # Record success history
                    S_phi2.append(Phi2)
                    delta_phi2.append(FitPop[idx2] - FitOff[idx2])
            else:
                Phi3 = np.clip(np.random.normal(phi3[idx_phi3], 0.1), 0.001, 0.5)
                Velocity[idx2] = np.random.rand(DimSize) * Velocity[idx2] + np.random.rand(DimSize) * (Pop[idx1] - Pop[idx2]) + Phi3 * Vec3
                Off[idx2] = Pop[idx2] + Velocity[idx2]
                Off[idx2] = Check(Off[idx2])
                FitOff[idx2] = func(Off[idx2])
                curFEs += 1
                if FitOff[idx2] < FitHis[idx2]:  # Individual update
                    FitHis[idx2] = FitOff[idx2]
                    His[idx2] = deepcopy(Off[idx2])
                if FitOff[idx2] < FitPop[idx2]:  # Record success history
                    S_phi3.append(Phi3)
                    delta_phi3.append(FitPop[idx2] - FitOff[idx2])


        else:
            Off[idx2] = deepcopy(Pop[idx2])
            FitOff[idx2] = FitPop[idx2]

            Vec1 = np.random.rand(DimSize) * (His[idx1] - Pop[idx1])
            Vec2 = np.random.rand(DimSize) * (Pop[np.argmin(FitPop)] - Pop[idx1])
            Vec3 = np.random.rand(DimSize) * (Xmean - Pop[idx1])

            r = np.random.rand()
            if r < 1 / 3:
                Phi1 = np.clip(np.random.normal(phi1[idx_phi1], 0.1), 0.001, 0.5)
                Velocity[idx1] = np.random.rand(DimSize) * Velocity[idx1] + np.random.rand(DimSize) * (Pop[idx2] - Pop[idx1]) + Phi1 * Vec1
                Off[idx1] = Pop[idx1] + Velocity[idx1]
                Off[idx1] = Check(Off[idx1])
                FitOff[idx1] = func(Off[idx1])
                curFEs += 1
                if FitOff[idx1] < FitHis[idx1]:
                    FitHis[idx1] = FitOff[idx1]
                    His[idx1] = deepcopy(Off[idx1])
                if FitOff[idx1] < FitPop[idx1]:
                    S_phi1.append(Phi1)
                    delta_phi1.append(FitPop[idx1] - FitOff[idx1])
            elif r < 2 / 3:
                Phi2 = np.clip(np.random.normal(phi2[idx_phi2], 0.1), 0.001, 0.5)
                Velocity[idx1] = np.random.rand(DimSize) * Velocity[idx1] + np.random.rand(DimSize) * (Pop[idx2] - Pop[idx1]) + Phi2 * Vec2
                Off[idx1] = Pop[idx1] + Velocity[idx1]
                Off[idx1] = Check(Off[idx1])
                FitOff[idx1] = func(Off[idx1])
                curFEs += 1
                if FitOff[idx1] < FitHis[idx1]:
                    FitHis[idx1] = FitOff[idx1]
                    His[idx1] = deepcopy(Off[idx1])
                if FitOff[idx1] < FitPop[idx1]:
                    S_phi2.append(Phi2)
                    delta_phi2.append(FitPop[idx1] - FitOff[idx1])
            else:
                Phi3 = np.clip(np.random.normal(phi3[idx_phi3], 0.1), 0.001, 0.5)
                Velocity[idx1] = np.random.rand(DimSize) * Velocity[idx1] + np.random.rand(DimSize) * (Pop[idx2] - Pop[idx1]) + Phi3 * Vec3
                Off[idx1] = Pop[idx1] + Velocity[idx1]
                Off[idx1] = Check(Off[idx1])
                FitOff[idx1] = func(Off[idx1])
                curFEs += 1
                if FitOff[idx1] < FitHis[idx1]:
                    FitHis[idx1] = FitOff[idx1]
                    His[idx1] = deepcopy(Off[idx1])
                if FitOff[idx1] < FitPop[idx1]:
                    S_phi3.append(Phi3)
                    delta_phi3.append(FitPop[idx1] - FitOff[idx1])

    c = 0.1
    if len(S_phi1) == 0:
        pass
    else:
        phi1[idx_phi1] = (1 - c) * phi1[idx_phi1] + c * meanWA(delta_phi1, S_phi1)
    if len(S_phi2) == 0:
        pass
    else:
        phi2[idx_phi2] = (1 - c) * phi2[idx_phi2] + c * meanWA(delta_phi2, S_phi2)
    if len(S_phi3) == 0:
        pass
    else:
        phi3[idx_phi3] = (1 - c) * phi3[idx_phi3] + c * meanWA(delta_phi3, S_phi3)

    Pop = deepcopy(Off)
    FitPop = deepcopy(FitOff)


def meanWA(delta, S):
    Delta = sum(delta)
    res = 0
    for i in range(len(S)):
        res += (delta[i] / Delta) * S[i]
    return res

def RunMSPCSO(func):
    global FitPop, curFEs, MaxFEs, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        BestList = []
        curFEs = 0
        np.random.seed(945 + 3 * i)
        Initialization(func)
        BestList.append(min(FitPop))
        while curFEs < MaxFEs:
            MSPCSO(func)
            BestList.append(min(FitPop))
        All_Trial_Best.append(BestList)
    np.savetxt("./MSPCSO_Data/Engineer/" + str(FuncNum) + ".csv", All_Trial_Best, delimiter=",")


def main():
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB

    MaxFEs = 10000
    MaxIter = int(MaxFEs / PopSize * 2)

    Probs = [CBD(), CBHD(), CSP(), GTD(), IBD(), PLD(), PVP(), RCB(), SRD(), TBTD(), TCD(), WBP()]
    Names = ["CBDP", "CBHDP", "CSDP", "GTDP", "IBDP", "PLDP", "PVDP", "RCBDP", "SRDP", "TBTDP", "TCDP", "WBDP"]

    for i in range(len(Probs)):
        DimSize = Probs[i].n_dims
        LB = Probs[i].lb
        UB = Probs[i].ub
        Pop = np.zeros((PopSize, DimSize))
        FuncNum = Names[i]
        RunMSPCSO(Probs[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('./MSPCSO_Data/Engineer') == False:
        os.makedirs('./MSPCSO_Data/Engineer')
    main()


