import numpy as np
import time
import pandas as pd
import torch
import os
import pickle
from torch_geometric.data import Data
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)

def normalizeDumb(s):
    s["geometry"] = s["geometry"].scale(xfact=2/1000, yfact=2/1000, origin=(0, 0))
    s["mean_x"] = s["mean_x"].apply(lambda e: e * 2/1000)
    s["mean_y"] = s["mean_y"].apply(lambda e: e * 2/1000)
    s["length"] = s["length"].apply(lambda e: e * 2/1000)
    s["geometry"] = s["geometry"].translate(xoff=-1, yoff=-1)
    s["mean_x"] = s["mean_x"].apply(lambda e: e - 1)
    s["mean_y"] = s["mean_y"].apply(lambda e: e - 1)
    return s

def normalizeDumbX(s):
    scale = np.vectorize(lambda e: e*2/1000)
    s[:,0:3] = scale(s[:,0:3])
    offset = np.vectorize(lambda e: e-1)
    s[:,0:2] = offset(s[:,0:2])
    s = s[:,[1,0,2,3]]
    return s

def adjMat2edgeIndex(a):
    edgeIndex = [[],[]]
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            if a[row][col]==1:
                edgeIndex[0].append(row)
                edgeIndex[1].append(col)
    return edgeIndex

def readLSV(file):
    lineInfo = []
    with open(file,"r") as f:
        for line in f:
            line = line.strip()
            lineA = line[:line.find("_")]
            lineB = line[line.find("_")+1:]
            lineInfo.append([int(lineA),int(lineB)])
    return lineInfo

graphs = {}
start = time.perf_counter()
for num,zi in enumerate(os.listdir("zinode/PaxHeader")):
    A = np.load("zinode/%s/A_%s.npy" % (zi,num)) #Adjacency matrix format
    A = torch.tensor(adjMat2edgeIndex(A)) #Edge index format
    X = np.load("zinode/%s/X_%s.npy" % (zi,num)) #Spatial features: x_min, y_min, l, θ
    X = torch.tensor(normalizeDumbX(X))
    S = torch.tensor(readLSV("zinode/%s/strokeid_%s.txt" % (zi,num))) #Temporal features: stroke i, line segment j
    N = torch.column_stack((X,S))
    print(N)
    graphs[zi] = Data(x=N, edge_index=A)
with open("graphsDictOrder.pickle","wb") as f:
    pickle.dump(graphs,f)
end = time.perf_counter()
print(end-start)