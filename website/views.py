from flask import Blueprint, render_template, request, redirect, url_for, session
from .models import Image
from . import db
import os
from skimage import data
import matplotlib.pyplot as plt
from skimage import filters
import numpy as np
import cv2
from skimage.morphology import skeletonize, thin
from skimage.util import invert
from .curveCalc import getJunctions, getEdgeEndPoints, getAdjacencyMatrix, getCurveData2, image_resize
from .hea import HEA

views = Blueprint('views',__name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['imgFile']
        path = os.path.join('.', 'img.jpg')
        file.save(path)
        session['calcState'] = request.form.get('calcState')
        return redirect(url_for('views.edit'))
    else:
        return render_template('home.html')

@views.route('/sample')
def sample():
    return render_template('sample.html')

@views.route('/edit')
def edit():
    originalImage = cv2.imread('img.jpg')
    originalImage = image_resize(originalImage, height = 400)
    height, width = originalImage.shape[:2]
    heightOffset = height/2 
    widthOffset = width/2 
    
    gray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2,2),np.uint8)

    (thresh, new) = cv2.threshold(blackAndWhiteImage, 1, 1, cv2.THRESH_BINARY)

    thinned = (thin(new).astype(np.uint8))*255


    kernel = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])/255
    img = cv2.filter2D(thinned, -1, kernel)
    img = img*(thin(new).astype(np.uint8))

    junctions = list()
    junctions = getJunctions(img)
    # for junction in junctions:
    #     for point in junction:
    #         cv2.circle(originalImage, point, 2, (255,0,0),-1)
    # cv2.imshow('img', originalImage)
    # cv2.waitKey(0)
    # print(junctions)
    endPoints = getEdgeEndPoints(junctions,img)
    matrix = getAdjacencyMatrix(junctions,endPoints,img)
    
    # for row in matrix:
    #     for column in row:
    #         print(column) 
    #         print('\n')
    #     print('next row')
    controlPoints = list()

    # print(matrix)

    # for row in matrix:
    #     for column in row:
    #         if column != 0:
    #             copy = originalImage
                
    #             for coord in column:
    #                 cv2.circle(copy, coord, 2, (255,0,0),-1)
    #             cv2.imshow('im', copy) 
    #             cv2.waitKey(0)

    for rowCount,nodes in enumerate(matrix):
        for columnCount,node in enumerate(nodes):
            if node != 0 and columnCount >= rowCount:
                path = node
                # print(path)
                curveData = getCurveData2(path)
                if len(curveData[0]) < 30:
                    continue
                # if len(curveData[0]) == 1:
                    # cv2.circle(originalImage, (curveData[0][0], curveData[1][0]), 4, (255,0,0),-1)
                    # cv2.imshow('img', originalImage)
                    # cv2.waitKey(0)
                results = HEA(curveData)
                results = results.flatten().tolist()
                controlPoints += results

    return render_template('edit.html', controlPoints=controlPoints)