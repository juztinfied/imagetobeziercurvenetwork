import random 
import math
from matplotlib import pyplot as plt 
import numpy as np

def mutate(x,sigma):
    print('x is', x)
    r1 = abs(np.random.normal(loc = 1,scale=0.1))
    r2 = abs(np.random.normal(loc = 1,scale=0.1))
    return x*np.array([[1],[r1],[r2],[1]])

def crossOver(xA, xB):
    rC = random.uniform(0,1.0)
    childA = rC*xA + (1-rC)*xB 
    childB = (1-rC)*xA + rC*xB 
    return (childA, childB)

def selectionOperator(probability):
    rS = random.uniform(0,1.00)

    for i in range(1,len(probability)+1):
        lowerBound = sum(probability[0:i-1]) 
        upperBound = sum(probability[0:i])
    
    if rS <= probability[0]: # probability[0] == p1
        return 0

    elif rS >= sum(probability[0:-2]):
        return (len(probability)-1)

    else:
        for i in range(1,len(probability)+1):
            lowerBound = sum(probability[0:i-1]) 
            upperBound = sum(probability[0:i])
            if lowerBound < rS and rS <= upperBound:
                return i 
    
    print('error in s(x)')
    return 

def fitness(xID, dataX, dataY):
    cp1 = xID[0]
    cp2 = xID[1]
    cp3 = xID[2]
    cp4 = xID[3]
    
    return (1/errorFunction(cp1,cp2,cp3,cp4,dataX,dataY))

def errorFunction(cp1,cp2,cp3,cp4,dataX,dataY):
    # print(cp1, cp2, cp3, cp4)
    if (len(dataX) != len(dataY)):
        print('dataX and dataY not same length')
        return 
    
    t = np.linspace(0,1,num=len(dataX))
    x = lambda t: ((1-t)**3)*cp1[0] + 3*t*((1-t)**2)*cp2[0] + 3*(t**2)*(1-t)*cp3[0] + (t**3)*cp4[0]
    y = lambda t: ((1-t)**3)*cp1[1] + 3*t*((1-t)**2)*cp2[1] + 3*(t**2)*(1-t)*cp3[1] + (t**3)*cp4[1]

    totalError = 0

    for k in range(0,len(dataX)):
        xError = (x(t)[k] - dataX[k])**2 
        yError = ((y(t)[k] - dataY[k])**2) 
        totalError += math.sqrt(xError + yError)
    
    # print(totalError)
    return totalError

def initiatePopulation(curveData):
    dataX = curveData[0]
    dataY = curveData[1]
    maxHeightXIncrease = curveData[2]
    maxHeightYIncrease = curveData[3]

    initialX = dataX[0]
    initialY = dataY[0]
    terminatingX = dataX[-1]
    terminatingY = dataY[-1]

    frontTangentDX = initialX - dataX[10]
    frontTangentDY = initialY - dataY[10]
    backTangentDX = terminatingX - dataX[-10]
    backTangentDY = terminatingY - dataY[-10]

    frontM = (dataY[10]-initialY)/(dataX[10]-initialX)
    frontC = initialY - frontM*initialX 

    backM = (dataY[-10]-terminatingY)/(dataX[-10]-terminatingX)
    backC = terminatingY - backM*terminatingX

    ratios = [0.75, 1, 1.33, 1.6]
    initialPop = list()


    for initialRatio in ratios:
        for terminatingRatio in ratios:
            lineM = ( (initialY+initialRatio*maxHeightYIncrease) - (terminatingY+terminatingRatio*maxHeightYIncrease) )/( (initialX+initialRatio*maxHeightXIncrease) - (terminatingX+terminatingRatio*maxHeightXIncrease) )
            lineC = (terminatingY+terminatingRatio*maxHeightYIncrease) - lineM*(terminatingX+terminatingRatio*maxHeightXIncrease)

            A = np.array([[-frontM,1],[-lineM,1]])
            B = np.array([frontC,lineC])
            cp2 = np.linalg.solve(A,B).tolist()

            A = np.array([[-backM,1],[-lineM,1]])
            B = np.array([backC,lineC])
            cp3 = np.linalg.solve(A,B).tolist()

            cp1 = [dataX[0], dataY[0]]
            cp4 = [dataX[-1], dataY[-1]]



            initialPop.append(np.array([cp1,cp2,cp3,cp4]))

    return initialPop

def pointGenerator(z,cp1,cp2,cp3,cp4):
    x = lambda t: ((1-t)**3)*cp1[0] + 3*t*((1-t)**2)*cp2[0] + 3*(t**2)*(1-t)*cp3[0] + (t**3)*cp4[0]
    y = lambda t: ((1-t)**3)*cp1[1] + 3*t*((1-t)**2)*cp2[1] + 3*(t**2)*(1-t)*cp3[1] + (t**3)*cp4[1]

    return x(z),y(z)


def HEA(curveData):
    dataX = curveData[0]
    dataY = curveData[1]
    maxHeightXIncrease = curveData[2]
    maxHeightYIncrease = curveData[3]

    t = 0
    T = 25
    w = 0.9
    c1 = c2 = 1.5 
    r = random.uniform(0.9,1.0)
    R = random.uniform(0.9,1.0)
    pR = 0.7
    pM = 0.05
    pC = 0.25
    sigma = 0.1*(math.sqrt(maxHeightXIncrease**2 + maxHeightYIncrease**2))

    z = np.linspace(0,1,num=len(dataX))

    popX = initiatePopulation(curveData) # this will generate a list, each element is a [cp1,cp2,cp3,cp4]
    
    popV = [None] * len(popX)
    popFitness = [fitness(xID,dataX,dataY) for xID in popX]
    pID = popX # list of the personal best position for each member of the population
    pND = popX[popFitness.index(max(popFitness))] # this should be a [cp1,cp2,cp3,cp4] which has the best fitness on record

    # print('best position ', pND)
    # cp1 = pND[0]
    # cp2 = pND[1]
    # cp3 = pND[2]
    # cp4 = pND[3]
    
    # plt.scatter(dataX, dataY,marker="s")
    # x,y = pointGenerator(z,cp1,cp2,cp3,cp4)
    # plt.scatter(x,y,marker="o")
    # plt.show()

    probability = [None] * len(popX)
    
    while t < T:
        # calculate V according to (1) and (4)
        for index,xID in enumerate(popX): # for each x position of each member of the population
            if popV[index] is None:
                popV[index] = c2*R*(pND-xID)
            else:
                popV[index] = w*popV[index] + c1*r*(pID[index]-xID) + c2*R*(pND-xID)

        # calculate X according to (2) and (5)
        for index,xID in enumerate(popX):
            # print('calculating new XID')
            newXID = xID + popV[index]

            # oldX,oldY = pointGenerator(z, xID[0], xID[1], xID[2], xID[3])
            # newX,newY = pointGenerator(z, newXID[0], newXID[1], newXID[2], newXID[3])

            # plt.scatter(dataX, dataY,color="b")
            # plt.scatter(oldX, oldY,color="r")
            # plt.scatter(newX, newY, color="g")
            # plt.show()

            if fitness(newXID,dataX,dataY) > fitness(xID,dataX,dataY): # if the new xID has better fitness, change the member's personal best position
                pID[index] = newXID
            
            popX[index] = newXID 
                
        # calculate F(X) according to (6) in P(t+1)
        popFitness = [fitness(xID,dataX,dataY) for xID in popX]
        
        # calculate p = (p1,p2,...,pN) according to (7)
        for index,fitnessI in enumerate(popFitness):
            probability[index] = fitnessI/sum(popFitness)
        
        
        newPopX = list() # prepare next generation of population
        newPopV = list()
        newPID = list()

        for k in range(0,len(probability),2):
            indexA = selectionOperator(probability)
            indexB = selectionOperator(probability)

            xA = popX[indexA]
            xB = popX[indexB]

            # print('xA and xB:\n')
            
            # x1,y1 = pointGenerator(z, xA[0], xA[1], xA[2], xA[3])
            # x2,y2 = pointGenerator(z, xB[0], xB[1], xB[2], xB[3])
            # plt.scatter(dataX, dataY,color="b")
            # plt.scatter(x1, y1,color="r")
            # plt.scatter(x2, y2,color="g")
            # plt.show()

            r = random.uniform(0,1.0)
            
            if r <= pR:
                newPopX.append(popX[indexA])
                newPopX.append(popX[indexB])
                newPopV.append(popV[indexA])
                newPopV.append(popV[indexB])
                newPID.append(pID[indexA])
                newPID.append(pID[indexB])
            
            elif r <= (pR + pC):
                childA,childB = crossOver(xA,xB)
                # print('childA and childB\n ')
                
                # x1,y1 = pointGenerator(z, childA[0], childA[1], childA[2], childA[3])
                # x2,y2 = pointGenerator(z, childB[0], childB[1], childB[2], childB[3])
                # plt.scatter(dataX, dataY,color="b")
                # plt.scatter(x1, y1,color="r")
                # plt.scatter(x2, y2,color="g")
                # plt.show()
                
                newPopX.append(childA)
                newPopX.append(childB)
                newPopV.append(None)
                newPopV.append(None)
                newPID.append(childA)
                newPID.append(childB)
            
            else:
                mutatedA = mutate(xA,sigma)
                mutatedB = mutate(xB,sigma)

                # print('mutated A and B: \n')
                
                # x1,y1 = pointGenerator(z, mutatedA[0], mutatedA[1], mutatedA[2], mutatedA[3])
                # x2,y2 = pointGenerator(z, mutatedB[0], mutatedB[1], mutatedB[2], mutatedB[3])
                # plt.scatter(dataX, dataY,color="b")
                # plt.scatter(x1, y1,color="r")
                # plt.scatter(x2, y2,color="g")
                # plt.show()

                newPopX.append(mutatedA)
                newPopX.append(mutatedB)
                newPopV.append(None)
                newPopV.append(None)
                newPID.append(mutatedA)
                newPID.append(mutatedB)
        
        
        popX = newPopX
        popV = newPopV
        pID = newPID
        
        popFitness = [fitness(member,dataX,dataY) for member in popX]
        probability = [None] * len(popFitness)

        for index,fitnessI in enumerate(popFitness):
            probability[index] = fitnessI/sum(popFitness)

        newPND = popX[popFitness.index(max(popFitness))] 
        
        if fitness(newPND,dataX,dataY) - fitness(pND,dataX,dataY) > 0 and fitness(newPND,dataX,dataY) - fitness(pND,dataX,dataY) < 0.000001 and fitness(newPND,dataX,dataY) > 0.0015:
            print('converging, hence early termination')
            # x,y = pointGenerator(z, newPND[0], pND[1], pND[2], pND[3])
            # plt.scatter(dataX, dataY,marker="s")
            # plt.scatter(x, y,marker="o")
            # plt.show()
            return newPND
        else:
            pND = popX[popFitness.index(max(popFitness))] 
            bestfit = max(popFitness)
            print('the best: ', bestfit)
            # x,y = pointGenerator(z, pND[0], pND[1], pND[2], pND[3])
            # plt.scatter(dataX, dataY,marker="s")
            # plt.scatter(x, y,marker="o")
            # plt.show()


        t += 1
    
    return pND