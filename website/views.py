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
from .curveCalc import getJunctions, getEdgeEndPoints, getAdjacencyMatrix, getCurveData2
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

@views.route('/edit')
def edit():
    originalImage = cv2.imread('img.jpg')
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
    endPoints = getEdgeEndPoints(junctions,img)
    matrix = getAdjacencyMatrix(junctions,endPoints,img)

    for rowCount,nodes in enumerate(matrix):
        for columnCount,node in enumerate(nodes):
            if node != 0 and columnCount >= rowCount:
                path = node
                curveData = getCurveData2(path)
                if rowCount == 0 and columnCount == 1:
                    results = HEA(curveData)
                    

    

    return render_template('edit.html')