import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from collections import defaultdict
from sklearn.linear_model import ElasticNetCV
import sys
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from math import sqrt

import contestLib as cl


###to mozna modyfikowac do woli
___poznanCenter = [52.407860, 16.928249]

___removalParams = [1, 1, 1, 1, 1, 1, 1, 1]


___learningDataBool = [
	1,	#rooms
	1,	#meters
	1,	#floors
	0,	#first coords
	0,	#second coords
	1	#distance
	]

___regression = linear_model.ElasticNetCV(
	l1_ratio=0.5, eps=0.0001, n_alphas=100, alphas=None,
	fit_intercept=True, normalize=False, precompute='auto',
	max_iter=10000, tol=0.0001, cv=None, copy_X=True,
	verbose=0, n_jobs=1, positive=False, random_state=None,
	selection='cyclic')

___predicted = 1 #0 is price, 1 is for price per meter
####



prices, rooms, meters, floors, pricesPerMeter = cl.loadTrainData()
firstCoords, secondCoords = cl.loadTrainCoords()

___distanceCenter = cl.getDistanceToPoint(firstCoords, secondCoords, ___poznanCenter[0], ___poznanCenter[1])


filteredData = cl.removeOutliers([prices, rooms, meters, floors, pricesPerMeter, firstCoords, secondCoords, ___distanceCenter], ___removalParams)

filtered_prices 			= filteredData[0]
filtered_rooms 				= filteredData[1]
filtered_meters 			= filteredData[2]
filtered_floors 			= filteredData[3]
filtered_pricesPerMeter 	= filteredData[4]
filtered_firstCoords 		= filteredData[5]
filtered_secondCoords 		= filteredData[6]
filtered_distanceCenter 	= filteredData[7]


dataList = [filtered_rooms, filtered_meters, filtered_floors, filtered_firstCoords, filtered_secondCoords, filtered_distanceCenter]


___learningData = list()

for i in range(len(dataList)):
	if ___learningDataBool[i] >= 1:
		___learningData.append(dataList[i])


scaler = cl.generateScaler(___learningData)
scaledData = cl.scaleData(___learningData, scaler)


expected = list()

if ___predicted >= 1:
	expected = filtered_pricesPerMeter
else:
	expected = filtered_prices


model = cl.createRegression(scaledData, expected, ___regression)


###########predykcja i ocena bledu

test_rooms, test_meters, test_floors = cl.loadDevData()
test_firstCoords, test_secondCoords = cl.loadDevCoords()
test_distanceCenter = cl.getDistanceToPoint(test_firstCoords, test_secondCoords, ___poznanCenter[0], ___poznanCenter[1])

test_dataList = [test_rooms, test_meters, test_floors, test_firstCoords, test_secondCoords, test_distanceCenter]
test_data = list()
for i in range(len(test_dataList)):
	if ___learningDataBool[i] >= 1:
		test_data.append(test_dataList[i])


scaled_testData = cl.scaleData(test_data, scaler)

predicted = cl.predict(scaled_testData, model)

if ___predicted >= 1:
	predicted = [a*b for a,b in zip(predicted,test_meters)]

test_expected = cl.loadDevExpected()
#for i in range(len(predicted)):
#	if predicted[i] < 250.0:
#		predicted[i] = predicted[i]*0.75

print(cl.rmse(test_expected, predicted))