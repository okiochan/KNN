import numpy as np
import data
import matplotlib.pyplot as plt
from sklearn import datasets

def euclidean(x,y):
    return np.sqrt(np.sum((x-y)**2))

def cntClasses(y):
    #n = Xl.shape[1]
    #y = Xl[:,n-1]
    return int(np.amax(y)+1)

#classes numbered from 0!
#takes xi from Xl, , Xl\xi, k
#returns best class
def KNN(x,Xl,Y,k):
    n = Xl.shape[0]
    dist = []
    for i in range (n) :
        dist.append(euclidean(x,Xl[i,:]))

    Xl = Xl[np.argsort (dist), : ]
    Y = Y[np.argsort (dist)]

    cnt = cntClasses(Y)
    score = [0]*cnt

    for i in range (k):
        curClass = int(Y[i])
        score[curClass] += 1

    return np.argmax(score)

#takes Xl,  K <= |Xl|
#returns best k
def LOO(Xl, Y, maxK):
    n = Xl.shape[0]

    error = [0]*maxK
    error[0] = 1000000

    for k in range (1,maxK) :
        for id in range (n-1) :
            x = Xl[id]
            y = int(Y[id])
            newXl = np.delete(Xl, id, axis=0)
            newY = np.delete(Y, id, axis=0)
            retClass = KNN(x,newXl,Y,k)

            if retClass != y :
                error[k] += 1

    bestK = np.argmin(error)
    return bestK


def classification_map(classifier, inp, out, xfrom=-2, xto=5, ticks=100):
    # meshgrid
    h = (xto - xfrom) / ticks
    xx, yy = np.arange(xfrom, xto, h), np.arange(xfrom, xto, h)
    xx, yy = np.meshgrid(xx, yy)
    zz = np.empty(xx.shape, dtype=float)

    # classify meshgrid
    pos = 0
    for x in range(xx.shape[0]):
        for y in range(yy.shape[0]):
            zz[x][y] = classifier(xx[x][y], yy[x][y])

    # display
    plt.clf()
    plt.contourf(xx, yy, zz, alpha=0.5) # class separations
    plt.scatter(inp[:,0], inp[:,1], c=out, s=50) # dataset points
    plt.show()



#----------------------------------------
iris = datasets.load_iris()
Xl = iris.data[:, [2,3]]
Y = iris.target
Xl,Y = data.getData()

#K = LOO(Xl,Y, Xl.shape[0])
#print(K)

K = 3

def ClassifyPoint(x, y):
    pt = np.array([x,y],dtype=float)
    curClass = KNN(pt,Xl,Y,K)
    return curClass

classification_map(ClassifyPoint, Xl, Y)

#plt.scatter(Xl[:,0], Xl[:,1], c=y, s=50)
#plt.show()
