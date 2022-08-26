import numpy as np
import time
import random
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import seaborn as sns
pd.set_option('display.max_columns', None)

def normalizeDumb(s):
    s["geometry"] = s["geometry"].scale(xfact=2/1000, yfact=2/1000, origin=(0, 0))
    s["mean_x"] = s["mean_x"].apply(lambda e: e * 2/1000)
    s["mean_y"] = s["mean_y"].apply(lambda e: e * 2/1000)
    s["length"] = s["length"].apply(lambda e: e * 2/1000)
    s["geometry"] = s["geometry"].translate(xoff=-1, yoff=-1)
    s["mean_x"] = s["mean_x"].apply(lambda e: e - 1)
    s["mean_y"] = s["mean_y"].apply(lambda e: e - 1)
    return s

def adjMat2edgeIndex(a):
    edgeIndex = [[],[]]
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            if a[row][col]==1:
                edgeIndex[0].append(row)
                edgeIndex[1].append(col)
    return edgeIndex

def normalizeDumbX(s):
    scale = np.vectorize(lambda e: e*2/1000)
    s[:,0:3] = scale(s[:,0:3])
    offset = np.vectorize(lambda e: e-1)
    s[:,0:2] = offset(s[:,0:2])
    s = s[:,[1,0,2,3]]
    return s

start = time.perf_counter()
for zi,num in [("草","6363")]:
    shape = gpd.read_file("zishp/%s/%s.shp" % (zi,num))
    shape = normalizeDumb(shape)
    #Plot line segments
    palette = sns.color_palette(None, len(shape)).as_hex()
    random.shuffle(palette)
    shape.plot(color=palette,linewidth=4)
    #Plot strokes
    palette = sns.color_palette(None,int(np.array(shape["stroke_id"])[-1][0])+1).as_hex()
    palette = [palette[int(_[0])] for _ in shape["stroke_id"]]
    shape.plot(color=palette,linewidth=4)
    #Plot character
    shape.plot(linewidth=4)
    plt.show()
    #Plot graph
    A = np.load("zinode/%s/A_%s.npy" % (zi,num))
    A = torch.tensor(adjMat2edgeIndex(A))
    X = np.load("zinode/%s/X_%s.npy" % (zi,num))
    X = torch.tensor(normalizeDumbX(X))
    graph = Data(x=X, edge_index=A)
    graphx = to_networkx(graph)
    pos = {i:(X[i,0],X[i,1]) for i in range(X.shape[0])}
    nx.draw(graphx,pos)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
end = time.perf_counter()