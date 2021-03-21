import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import skeletonize, thin
from skimage.util import invert
import math

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def isEightConnected(x1,y1,x2,y2): 
    if abs(x1-x2) == 0 or abs(x1-x2) == 1:
        if abs(y1-y2) == 0 or abs(y1-y2) == 1:
            return True 
    return False 

def addToJunctions(junctions,x,y): 
    for junction in junctions:
        for pixel in junction:
            pixelX,pixelY = pixel 
            if isEightConnected(pixelX,pixelY,x,y):
                junction.append((x,y))
                return junctions
    
    junctions.append([(x,y)])
    return junctions

def getJunctions(neighbourMatrix): 
    junctions = list()
    height, width = neighbourMatrix.shape
    for y in range(0,height):
        for x in range (0,width):
            if neighbourMatrix.item(y,x) >= 3:
                junctions = addToJunctions(junctions,x,y)
    
    return junctions 

def getEdgeEndPoints(junctions,neighbourMatrix):  
    endPoints = list()
    height, width = neighbourMatrix.shape
    for y in range(0,height):
        for x in range (0,width):
            if neighbourMatrix.item(y,x) == 2:
                for junctionIndex,junction in enumerate(junctions):
                    for pixelX,pixelY in junction:
                        if isEightConnected(pixelX,pixelY,x,y):
                            endPoints.append((x,y,junctionIndex))
                
    
    return endPoints

def findEightPath(endPoints, x,y,path,neighbourMatrix):
    for Y in range(y-1,y+2):
        for X in range(x-1,x+2):
            if Y == y and X == x:
                continue 
            elif neighbourMatrix.item(Y,X) == 2:
                if (X,Y) not in path:
                    for endX,endY,junctionIndex in endPoints:
                        if endX == X and endY == Y:
                            path.append((endX,endY))
                            return path  
                    
                    path.append((X,Y))
                    path = findEightPath(endPoints, X,Y,path,neighbourMatrix)

    return path 

def getAdjacencyMatrix(junctions,endPoints,neighbourMatrix):
    matrix = [ [0] * len(junctions) for _ in range(len(junctions))]
    for endX,endY,junctionIndex in endPoints:
        path = findEightPath(endPoints, endX,endY,[(endX,endY)],neighbourMatrix)
        start = path[0]
        startJunctionIndex = -1
        end = path[-1]
        endJunctionIndex = -1

        for x, y, index in endPoints:
            if (x,y) == start:
                startJunctionIndex = index 
            if (x,y) == end:
                endJunctionIndex = index 
        if startJunctionIndex != -1 and endJunctionIndex != -1:
            matrix[startJunctionIndex][endJunctionIndex] = path  
    
    return matrix 

def getCurveData2(path):
    # first get line equation of the line connecting the end points
    startX,startY = path[0]
    endX,endY = path[-1]

    dataX = list() 
    dataY = list()
    maxHeightX = maxHeightY = maxHeightIncrease = -1

    if startX == endX: # means the baseline is a vertical line going up/down
        maxHeightY = 0
        for x,y in path:
            dataX.append(x)
            dataY.append(y) 
            distance = abs(x-startX)
            if distance >= maxHeightIncrease:
                maxHeightIncrease = distance
                maxHeightX = x
        
        maxHeightXIncrease = maxHeightX - startX
        maxHeightYIncrease = 0 
        return [dataX, dataY, maxHeightXIncrease, maxHeightYIncrease]
    
    elif startY == endY: # means the base line is a flat horizontal line
        maxHeightX = 0
        for x,y in path:
            dataX.append(x)
            dataY.append(y) 
            distance = abs(y-startY)
            if distance >= maxHeightIncrease:
                maxHeightIncrease = distance
                maxHeightY = y
        
        maxHeightXIncrease = 0
        maxHeightYIncrease = maxHeightY - startY
        return [dataX, dataY, maxHeightXIncrease, maxHeightYIncrease]
                
    else:
        baseLineM = (endY-startY)/(endX-startX)
        baseLineC = startY - baseLineM*startX 
        
        A = startY - endY # y1 - y2
        B = endX - startX # x2 - x1
        C = startX*endY - endX*startY # x1y2 - x2y1
        
        # then for each point in the path, calculate its perpendicular distance to that line
        
        for x,y in path:
            dataX.append(x)
            dataY.append(y) 
            distance = abs((A*x + B*y + C)) / math.sqrt(A**2 + B**2)
            if distance >= maxHeightIncrease:
                maxHeightIncrease = distance
                maxHeightX = x
                maxHeightY = y
        
        # find the point on the line that is closest to the point of the path
        
        # find vector of projection line
        projX = maxHeightX - startX 
        projY = maxHeightY - startY 

        # find vector of base line
        baseX = endX - startX 
        baseY = endY - startY 

        # find length of projection vector 
        projMag = math.sqrt((maxHeightX - startX)**2 + (maxHeightY - startY)**2)
        baseMag = math.sqrt((endX - startX)**2 + (endY - startY)**2)

        # find dot product 
        dotProduct = projX*baseX + projY*baseY 

        # find length of projection on base line
        length = dotProduct/projMag 
        fraction = length/baseMag 

        pointX = int(startX + fraction*(endX-startX))
        pointY = int(startY + fraction*(endY-startY))

        maxHeightXIncrease = maxHeightX - pointX 
        maxHeightYIncrease = maxHeightY - pointY 

        return [dataX, dataY, maxHeightXIncrease, maxHeightYIncrease]


def approxCPS(curveData):
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
    error = finalCP1 = finalCP2 = finalCP3 = finalCP4 = None

    for initialRatio in ratios:
        for terminatingRatio in ratios:
            lineM = ( (initialY+initialRatio*maxHeightYIncrease) - (terminatingY+terminatingRatio*maxHeightYIncrease) )/( (initialX+initialRatio*maxHeightXIncrease) - (terminatingX+terminatingRatio*maxHeightXIncrease) )
            lineC = (terminatingY+terminatingRatio*maxHeightYIncrease) - lineM*(terminatingX+terminatingRatio*maxHeightXIncrease)

            A = np.array([[-frontM,1],[-lineM,1]])
            B = np.array([frontC,lineC])
            cp2 = np.linalg.solve(A,B)

            A = np.array([[-backM,1],[-lineM,1]])
            B = np.array([backC,lineC])
            cp3 = np.linalg.solve(A,B)

            cp1 = (dataX[0], dataY[0])
            cp4 = (dataX[-1], dataY[-1])

            if error == None:
                error = errorFunction(cp1, cp2, cp3, cp4, dataX, dataY)
                print('first', error)
            elif errorFunction(cp1, cp2, cp3, cp4, dataX, dataY) < error:
                error = errorFunction(cp1, cp2, cp3, cp4, dataX, dataY)
                print('renewed', error)
                finalCP1 = cp1
                finalCP2 = cp2 
                finalCP3 = cp3
                finalCP4 = cp4 
    
    print(finalCP1,finalCP2,finalCP3,finalCP4)
    return (finalCP1,finalCP2,finalCP3,finalCP4)


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

