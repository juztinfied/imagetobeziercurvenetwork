import random 
import math
from matplotlib import pyplot as plt 
import numpy as np

def mutate(x,sigma):
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

def fitness(xI, dataX, dataY):
    cp1 = xI[0]
    cp2 = xI[1]
    cp3 = xI[2]
    cp4 = xI[3]
    
    return (1/errorFunction(cp1,cp2,cp3,cp4,dataX,dataY))

def errorFunction(cp1,cp2,cp3,cp4,dataX,dataY):
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

    frontM = frontC = backM = backC = None
    
    if (dataX[10] == initialX):
        frontM = (dataY[10]-initialY)
    else:
        frontM = (dataY[10]-initialY)/(dataX[10]-initialX)
    
    frontC = initialY - frontM*initialX 

    if (dataX[-10] == terminatingX):
        backM = (dataY[-10]-terminatingY)
    else:
        backM = (dataY[-10]-terminatingY)/(dataX[-10]-terminatingX)

    backC = terminatingY - backM*terminatingX

    ratios = [0.75, 1, 1.33, 1.6]
    initialPop = list()

    print(frontM)
    print(frontC)
    print(backM)
    print(backC)


    for initialRatio in ratios:
        for terminatingRatio in ratios:
            if maxHeightXIncrease == 0: # means base line is a flat horizontal line
                lineC = initialY+initialRatio*maxHeightYIncrease
                A = np.array([[-frontM,1],[0,1]])
                B = np.array([frontC,lineC])
                cp2 = np.linalg.solve(A,B).tolist()

                lineC = terminatingY+terminatingRatio*maxHeightYIncrease
                A = np.array([[-backM,1],[0,1]])
                B = np.array([backC,lineC])
                cp3 = np.linalg.solve(A,B).tolist()

                cp1 = [dataX[0], dataY[0]]
                cp4 = [dataX[-1], dataY[-1]]

                initialPop.append(np.array([cp1,cp2,cp3,cp4]))

            elif maxHeightYIncrease == 0: # means base line is a straight up vertical line
                cp1 = [dataX[0], dataY[0]]
                cp4 = [dataX[-1], dataY[-1]]

                lineX = initialX+initialRatio*maxHeightXIncrease 
                newX = frontM*lineX + frontC 
                cp2 = [newX, initialY]

                lineX = terminatingX+terminatingRatio*maxHeightXIncrease 
                newX = backM*lineX + backC 
                cp3 = [newX, terminatingY]

                

                initialPop.append(np.array([cp1,cp2,cp3,cp4]))
            
            else: 
                cp1 = [dataX[0], dataY[0]]
                cp4 = [dataX[-1], dataY[-1]]
                lineM = ( (initialY+initialRatio*maxHeightYIncrease) - (terminatingY+terminatingRatio*maxHeightYIncrease) )/( (initialX+initialRatio*maxHeightXIncrease) - (terminatingX+terminatingRatio*maxHeightXIncrease) )
                lineC = (terminatingY+terminatingRatio*maxHeightYIncrease) - lineM*(terminatingX+terminatingRatio*maxHeightXIncrease)

                A = np.array([[-frontM,1],[-lineM,1]])
                B = np.array([frontC,lineC])

                if frontM == lineM:
                    cp2 = cp1 
                else:
                    cp2 = np.linalg.solve(A,B).tolist()

                A = np.array([[-backM,1],[-lineM,1]])
                B = np.array([backC,lineC])
   
                if backM == lineM:
                    cp3 = cp4 
                else:
                    cp3 = np.linalg.solve(A,B).tolist()

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
    T = 2
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
    popFitness = [fitness(xI,dataX,dataY) for xI in popX]
    pI = popX # list of the personal best position for each member of the population
    pG = popX[popFitness.index(max(popFitness))] # this should be a [cp1,cp2,cp3,cp4] which has the best fitness on record
    bestInHistory = pG


    probability = [None] * len(popX)
    
    while t < T:
        # calculate V according to (1) and (4)
        for index,xI in enumerate(popX): # for each x position of each member of the population
            if popV[index] is None:
                popV[index] = c2*R*(pG-xI)
            else:
                popV[index] = w*popV[index] + c1*r*(pI[index]-xI) + c2*R*(pG-xI)

        # calculate X according to (2) and (5)
        for index,xI in enumerate(popX):
            newXI = xI + popV[index]

            if fitness(newXI,dataX,dataY) > fitness(xI,dataX,dataY): # if the new xI has better fitness, change the member's personal best position
                pI[index] = newXI
            
            popX[index] = newXI 
                
        # calculate F(X) according to (6) in P(t+1)
        popFitness = [fitness(xI,dataX,dataY) for xI in popX]
        
        # calculate p = (p1,p2,...,pN) according to (7)
        for index,fitnessI in enumerate(popFitness):
            probability[index] = fitnessI/sum(popFitness)
        
        
        newPopX = list() # prepare next generation of population
        newPopV = list()
        newPI = list()

        for k in range(0,len(probability),2):
            indexA = selectionOperator(probability)
            indexB = selectionOperator(probability)

            xA = popX[indexA]
            xB = popX[indexB]

            r = random.uniform(0,1.0)
            
            if r <= pR:
                newPopX.append(popX[indexA])
                newPopX.append(popX[indexB])
                newPopV.append(popV[indexA])
                newPopV.append(popV[indexB])
                newPI.append(pI[indexA])
                newPI.append(pI[indexB])
            
            elif r <= (pR + pC):
                childA,childB = crossOver(xA,xB)
                
                newPopX.append(childA)
                newPopX.append(childB)
                newPopV.append(None)
                newPopV.append(None)
                newPI.append(childA)
                newPI.append(childB)
            
            else:
                mutatedA = mutate(xA,sigma)
                mutatedB = mutate(xB,sigma)

                newPopX.append(mutatedA)
                newPopX.append(mutatedB)
                newPopV.append(None)
                newPopV.append(None)
                newPI.append(mutatedA)
                newPI.append(mutatedB)
        
        
        popX = newPopX
        popV = newPopV
        pI = newPI
        
        popFitness = [fitness(member,dataX,dataY) for member in popX]
        probability = [None] * len(popFitness)

        for index,fitnessI in enumerate(popFitness):
            probability[index] = fitnessI/sum(popFitness)

        newPG = popX[popFitness.index(max(popFitness))] 
        
        if fitness(newPG,dataX,dataY) - fitness(pG,dataX,dataY) > 0 and fitness(newPG,dataX,dataY) - fitness(pG,dataX,dataY) < 0.000001 and fitness(newPG,dataX,dataY) > 0.0015 and fitness(newPG, dataX, dataY) > fitness(bestInHistory, dataX, dataY):
            print('converging, hence early termination')
            return newPG

        else:
            pG = popX[popFitness.index(max(popFitness))] 
            bestfit = max(popFitness)

            if fitness(pG,dataX,dataY) > fitness(bestInHistory,dataX,dataY):
                bestInHistory = pG



        t += 1

    print('the best in history for this curve: ', fitness(bestInHistory,dataX,dataY))
    return bestInHistory