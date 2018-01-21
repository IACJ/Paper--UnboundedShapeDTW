# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:51:56 2017

@author: IACJ
"""
from os.path import expanduser
from numpy import genfromtxt
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from numpy import array, zeros, argmin, inf
import collections

class US_DTW(object):
    
    
    def __init__(self,x,y):
        
   
        self.x = x
        self.y = y

        
        self.resultDistance = None

        self.sp = None
        
#        self.threshold = 0.1
        self.abandonLengthRatio = 0.3
        self.plot = True
        self.showDetails = True
        self.descriptors = "None"
        self.descriptors_l = 6
    
   
        self.Init_AllPoints()
        self.forward()
        self.backward()
        self.findTrace()
        self.describePaths()


    def describePaths(self):
        if self.showDetails:
            print("Paths :",len(self.paths))
        
        accPunish = 0
        accLen = 0
        for dx,dy,punish in self.paths:
            if self.showDetails:
                print(len(dx),punish,punish / len(dx))
            accLen += len(dx)
            accPunish += punish
        
        if accLen != 0 :
            self.resultDistance = accPunish /accLen
        else :
            self.resultDistance = inf
        if self.showDetails :
            print("resultDistance : ",self.resultDistance)
        
    def showResult(self, step=1):
 
        p0 = np.array(list(self.path[0]),dtype=int)
        p1 = np.array(list(self.path[1]),dtype=int)
        

        plt.imshow(self.cost, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
        plt.plot(p1, p0, '-ro') # relation
        plt.title('Warping Matrix')
        plt.show()
        
#         绘制对齐图
        x = self.x.copy()
        y = self.y.copy() + x.max()
            
        plt.plot(x)
        plt.plot(y)
    
        #绘制对齐线
        for i in range(0,len(p1),step):
            plt.plot([p0[i],p1[i]],  [x[p0[i]],y[p1[i]]],'y')
            
        plt.title('Dynamic Time Warping')
        plt.grid(True)
        plt.show()       
        print("如上图所示")
        
    def Init_AllPoints(self):
        
        self.sp = np.zeros((len(self.x),len(self.y)))
        self.dis = np.ones((len(self.x),len(self.y))) * inf
        
        if self.descriptors == "None":
            for i in range (0, len(self.x)):
                for j in range(0, len(self.y)):
                    self.sp[i][j] = 1;
                    self.dis[i][j] = abs(self.x[i] - self.y[j])
            
        elif self.descriptors == "PAA":
            self.descriptorsX = self.x.copy()
            self.descriptorsY = self.y.copy()
            for i in range (0, len(self.x)):
                self.descriptorsX[i] = (self.x[max(0,i-self.descriptors_l//2):min(len(self.x),i+self.descriptors_l//2)]).mean()
            for j in range (0, len(self.y)):
                self.descriptorsY[j] = (self.y[max(0,j-self.descriptors_l//2):min(len(self.y),j+self.descriptors_l//2)]).mean()
            for i in range (0, len(self.x)):
                for j in range(0, len(self.y)):
                    self.sp[i][j] = 1;
                    self.dis[i][j] = abs(self.descriptorsX[i] - self.descriptorsY[j])                
                

        
        self.threshold = 0.6*self.dis[0:len(self.x)][0:len(self.y)].mean()
        print(self.threshold)
        self.D = self.dis.copy()
        self.M = np.ones((len(self.x)+1,len(self.y)+1),dtype = int) 
        self.traceForward = np.ones((len(self.x),len(self.y),2),dtype = int) * (-1)
        self.traceBackward = np.ones((len(self.x),len(self.y),2),dtype = int) * (-1)

        
    def forward(self):
        
        for n in range (0, len(self.x)-1):
            for m in range(0, len(self.y)-1):
                if (self.sp[n][m] ==1 or self.sp[n][m] ==3):
                    i,j = 1,0
                    if ( (self.D[n][m]+self.dis[n+i][m+j])/(self.M[n,m]+1) <= self.D[n+i,m+j] /self.M[n+i][m+j]
                            and (self.D[n][m]+self.dis[n+i][m+j])/(self.M[n,m]+1) <= self.threshold ):
                        self.M[n+i][m+j] = self.M[n][m] +1
                        self.D[n+i][m+j] = self.D[n][m]+ self.dis[n+i][m+j]
                        self.sp[n+i][m+j] = 3
                        self.traceForward[n+i][m+j] = n,m
                      
                                       
                    i,j = 0,1
                    if ( (self.D[n][m]+self.dis[n+i][m+j])/(self.M[n,m]+1) <= self.D[n+i,m+j] /self.M[n+i][m+j]
                            and (self.D[n][m]+self.dis[n+i][m+j])/(self.M[n,m]+1) <= self.threshold ):
                        self.M[n+i][m+j] = self.M[n][m] +1
                        self.D[n+i][m+j] = self.D[n][m]+ self.dis[n+i][m+j]
                        self.sp[n+i][m+j] = 3
                        self.traceForward[n+i][m+j] = n,m

                    i,j = 1,1
                    if ( (self.D[n][m]+2*self.dis[n+i][m+j])/(self.M[n,m]+2) <= self.D[n+i,m+j] /self.M[n+i][m+j]
                            and (self.D[n][m]+2*self.dis[n+i][m+j])/(self.M[n,m]+2) <= self.threshold ):
                        self.M[n+i][m+j] = self.M[n][m] +2
                        self.D[n+i][m+j] = self.D[n][m]+ 2*self.dis[n+i][m+j]
                        self.sp[n+i][m+j] = 3
                        self.traceForward[n+i][m+j] = n,m
                 
        if self.plot :
            plt.imshow(self.sp, origin='lower', cmap="BuPu_r", interpolation='nearest')
            plt.title('Forward')
            plt.show()
        
    def backward(self):

        for n in range (len(self.x)-1, 0, -1):
            for m in range(len(self.y)-1, 0 ,-1):
                if (self.sp[n][m] ==1 or self.sp[n][m] ==2):

                    i,j = -1,0
                    if ( (self.D[n][m]+self.dis[n+i][m+j])/(self.M[n,m]+1) <= self.D[n+i,m+j] /self.M[n+i][m+j]
                            and (self.D[n][m]+self.dis[n+i][m+j])/(self.M[n,m]+1) <= self.threshold ):
                        self.M[n+i][m+j] = self.M[n][m] +1
                        self.D[n+i][m+j] = self.D[n][m]+ self.dis[n+i][m+j]
                        self.sp[n+i][m+j] = 2
                        self.traceBackward[n+i][m+j] = n,m
             
                    
                    i,j = 0,-1
                    if ( (self.D[n][m]+self.dis[n+i][m+j])/(self.M[n,m]+1) <= self.D[n+i,m+j] /self.M[n+i][m+j]
                            and (self.D[n][m]+self.dis[n+i][m+j])/(self.M[n,m]+1) <= self.threshold ):
                        self.M[n+i][m+j] = self.M[n][m] +1
                        self.D[n+i][m+j] = self.D[n][m]+ self.dis[n+i][m+j]
                        self.sp[n+i][m+j] = 2
                        self.traceBackward[n+i][m+j] = n,m
              

                    i,j = -1,-1
                    if ( (self.D[n][m]+2*self.dis[n+i][m+j])/(self.M[n,m]+2) <= self.D[n+i,m+j] /self.M[n+i][m+j]
                            and (self.D[n][m]+2*self.dis[n+i][m+j])/(self.M[n,m]+2) <= self.threshold ):
                        self.M[n+i][m+j] = self.M[n][m] +2
                        self.D[n+i][m+j] = self.D[n][m]+ 2*self.dis[n+i][m+j]
                        self.sp[n+i][m+j] = 2
                        self.traceBackward[n+i][m+j] = n,m
                    
        if self.plot :                        
            plt.imshow(self.sp, origin='lower', cmap="BuPu_r", interpolation='nearest')
            plt.title('Backward')
            plt.show()
        
         
    def findTrace(self):
        
        self.goddessHead =  np.ones((len(self.x),len(self.y)),dtype = int) * (-1) 
        self.goddessTail =  np.ones((len(self.x),len(self.y)),dtype = int) * (-1) 
        
        
        forwardSteps = self.M.copy()
        for n in range (len(self.x)-1, -1, -1):
            for m in range(len(self.y)-1, -1 ,-1):
                if (self.traceForward[n][m][0] != -1):
                    i,j = self.traceForward[n][m]
                    if(forwardSteps[i][j] <= forwardSteps[n][m]):
                        forwardSteps[i][j] = forwardSteps[n][m]
                        if (self.sp[i][j] == 1):
                            self.goddessHead[i][j] = forwardSteps[i][j]
                        else :
                            self.sp[i][j] = 100
                          
                        
                    
        backwardSteps = self.M.copy()  
        for n in range (0, len(self.x)):
            for m in range(0, len(self.y)):
                if (self.traceBackward[n][m][0] != -1):
                    i,j = self.traceBackward[n][m]
                    if(backwardSteps[i][j] <= backwardSteps[n][m]):
                        backwardSteps[i][j] = backwardSteps[n][m]               
                        if (self.sp[i][j] == 1):
                            self.goddessTail[i][j] = backwardSteps[i][j]
                        else:
                            self.sp[i][j] = 100
        if self.plot :                            
            plt.imshow(self.sp, origin='lower', cmap="BuPu_r", interpolation='nearest')
            plt.title('BackTrace')
            plt.show() 
                            
        self.goddessTotal = self.goddessHead+self.goddessTail
        
        for n in range (0, len(self.x)):
            for m in range(0, len(self.y)):
        
                if (self.goddessTotal[n][m] < (len(self.x) *2 * self.abandonLengthRatio)) :
                    self.goddessTotal[n][m] = -1
                    self.sp[n][m] = 0
                else :
                    if self.showDetails:
                        print(n,m,self.goddessHead[n][m],self.goddessTail[n][m],self.goddessTotal[n][m])
            
        if self.plot :        
            plt.imshow(self.sp, origin='lower', cmap="BuPu_r", interpolation='nearest')
            plt.title('Goddess')
            plt.show()   
        
        self.paths= []
        for n in range (0, len(self.x)):
            for m in range(0, len(self.y)):
                if  (self.sp[n][m] == 1):
                
                    dx = collections.deque()
                    dy = collections.deque()
                    dx.append(n)
                    dy.append(m)
                    nown = n
                    nowm = m
                    punish = self.dis[nown][nowm]
                    while(True):
                        i,j = 1,1 
                        if (forwardSteps[nown+i][nowm+j] == self.goddessHead[n][m]):
                            dx.append(nown+i)
                            dy.append(nowm+j)
                            self.sp[nown+i][nowm+j] = 4     
                            nown,nowm = nown+i,nowm+j
                            punish = self.dis[nown][nowm]
                            continue
                        
                        i,j = 1,0  
                        if (forwardSteps[nown+i][nowm+j] == self.goddessHead[n][m]):
                            dx.append(nown+i)
                            dy.append(nowm+j)
                            self.sp[nown+i][nowm+j] = 4     
                            nown,nowm = nown+i,nowm+j
                            punish = self.dis[nown][nowm]
                            continue
                        i,j = 0,1  

                        if (forwardSteps[nown+i][nowm+j] == self.goddessHead[n][m]):
                            dx.append(nown+i)
                            dy.append(nowm+j)
                            self.sp[nown+i][nowm+j] = 4     
                            nown,nowm = nown+i,nowm+j
                            punish = self.dis[nown][nowm]
                            continue

                        break
                    nown = n
                    nowm = m
#                    while (nown>0 and nowm>0):
                    while(True):
                        i,j = -1,-1  
                        if (backwardSteps[nown+i][nowm+j] == self.goddessTail[n][m]):
                            dx.appendleft(nown+i)
                            dy.appendleft(nowm+j)
                            self.sp[nown+i][nowm+j] = 4     
                            nown,nowm = nown+i,nowm+j
                            punish = self.dis[nown][nowm]
                            continue
                        i,j = -1,0  
                        if (backwardSteps[nown+i][nowm+j] == self.goddessTail[n][m]):
                            dx.appendleft(nown+i)
                            dy.appendleft(nowm+j)
                            self.sp[nown+i][nowm+j] = 4     
                            nown,nowm = nown+i,nowm+j
                            punish = self.dis[nown][nowm]
                            continue
                        i,j = 0,-1  
                        if (backwardSteps[nown+i][nowm+j] == self.goddessTail[n][m]):
                            dx.appendleft(nown+i)
                            dy.appendleft(nowm+j)
                            self.sp[nown+i][nowm+j] = 4     
                            nown,nowm = nown+i,nowm+j
                            punish = self.dis[nown][nowm]
                            continue

                        break
                    self.paths.append((dx,dy,punish) )
                    
        if self.plot :
            plt.imshow(self.sp, origin='lower', cmap="BuPu_r", interpolation='nearest')
            plt.title('Paths')
            plt.show()  
        
           
#    def dis(self,i,j):       
#        if (i>=len(self.x) or j>=len(self.y)):
#            return inf
#        
#        if (self.distance[i][j] == -1):
#            self.distance[i][j] = self.dist(self.x[i],self.y[j])
#        return self.distance[i][j]



# 别人的DTW代码，直接拷贝进来，省去import和pip install的麻烦
def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path
# 别人的DTW代码，直接拷贝进来，省去import和pip install的麻烦
def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)




def showRowData(x,y):
    plt.plot(x)
    plt.plot( y + x.max() )
    plt.title('Raw Time Series')
    plt.grid(True)
    plt.show()

# 绘图函数
def show(xx,yy,path,step=1):
    '用于绘图的函数'
    
    x = np.array(xx)
    y = np.array(yy)
    p0 = np.array(list(path[0]),dtype=int)
    p1 = np.array(list(path[1]),dtype=int)
    
    #绘制矩阵图
    plt.imshow(cost, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    plt.plot(p1, p0, '-ro') # relation

#    plt.axis('tight')
    plt.show()
    
    # 绘制对齐图
    x = x-10
    xIndex = np.zeros(len(x))
    for i in range(len(x)):
        xIndex[i] = i;
        
    
    plt.plot(x)
    plt.plot(y)

    #绘制对齐线
    for i in range(0,len(p1),step):
        plt.plot([p0[i], p1[i]],  [x[p0[i]],  y[ p1[i] ] ],'y')
        
    plt.title('Dynamic Time Warping')
    plt.grid(True)
    plt.show()
  


if __name__ == '__main__':    

    print("program begin ")


    
    ##### 参数与超参数： #########
    n = 50        #时间序列长度
    randInt =50    #随机生成的时间序列数值范围
    distance = euclidean_distances  # 利用欧式距离
    
    separationDistance = 1  # 关键点之间的间距
    abandonLengthRatio = 0.5 # 路径要超过x的多少倍才不会被舍弃
    threshold = 1  # 平均惩罚值必须要小于的值
    
    #############################
    x=[]
    for i in range(n):
        x.append(np.random.randint(10))
    x = np.array(x)
    y = np.tile(x,3)
    
#    showRowData(x,y)
#    dist, cost, acc, path = dtw(x, y, euclidean_distances)
#    show(x,y,path)
#    print('DTW = ',acc[-1][-1])
    
    us_dtw = US_DTW(x,y)
    print("Paths :",len(us_dtw.paths))
    print(us_dtw.resultDistance)
  
    