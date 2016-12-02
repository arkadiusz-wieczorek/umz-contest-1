#  -*- coding: utf-8 -*-

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


def rmse(predicted, actual):
    return np.sqrt(((predicted - actual) ** 2).mean())


def loadTrainData():
	data = list()
	with open('../train/train.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			data.append(row)

	dataZipped = zip(*data)
	columns = list(dataZipped)

	prices = np.array(list(map(float, columns[0])))
	rooms = np.array(list(map(float, columns[1])))
	meters = np.array(list(map(float, columns[2])))
	floors = np.array(list(map(float, columns[3])))
	pricesPerMeter = [a/b for a,b in zip(prices,meters)]
	return prices, rooms, meters, floors, pricesPerMeter


def loadTrainCoords():
	#reading coords
	coordData = list()
	with open('../train/coords.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			coordData.append(row)

	columnsZipped = zip(*coordData)
	coordColumns = list(columnsZipped);

	firstCoords = np.array(list(map(float, coordColumns[0])))
	secondCoords = np.array(list(map(float, coordColumns[1])))
	return firstCoords, secondCoords


def loadDevData():
	data = list()
	with open('../dev-0/in.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			data.append(row)

	dataZipped = zip(*data)
	columns = list(dataZipped)

	rooms = np.array(list(map(float, columns[0])))
	meters = np.array(list(map(float, columns[1])))
	floors = np.array(list(map(float, columns[2])))
	return rooms, meters, floors


def loadDevCoords():
	#reading coords
	coordData = list()
	with open('../dev-0/coords.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			coordData.append(row)

	columnsZipped = zip(*coordData)
	coordColumns = list(columnsZipped);

	firstCoords = np.array(list(map(float, coordColumns[0])))
	secondCoords = np.array(list(map(float, coordColumns[1])))
	return firstCoords, secondCoords

def loadDevExpected():
	data = list()
	with open('../dev-0/expected.tsv','r', encoding="utf8") as file:
		for row in file:
			data.append(row)

	data2 = np.array(list(map(float, data)))
	return data2
	
def loadTestData():
	data = list()
	with open('../test-A/train.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			data.append(row)

	dataZipped = zip(*data)
	columns = list(dataZipped)

	rooms = np.array(list(map(float, columns[0])))
	meters = np.array(list(map(float, columns[1])))
	floors = np.array(list(map(float, columns[2])))
	return rooms, meters, floors


def loadTestCoords():
	#reading coords
	coordData = list()
	with open('../test-A/coords.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			coordData.append(row)

	columnsZipped = zip(*coordData)
	coordColumns = list(columnsZipped);

	firstCoords = np.array(list(map(float, coordColumns[0])))
	secondCoords = np.array(list(map(float, coordColumns[1])))
	return firstCoords, secondCoords


def getDistanceToPoint(firstCoords, secondCoords, xb, yb):
	distanceCenter = list()

	for i in range(len(firstCoords)):
		xa = firstCoords[i]
		ya = secondCoords[i]
		dist = np.sqrt((xa-xb)**2 + (ya-yb)**2)
		distanceCenter.append(dist)
	return distanceCenter


def _rejectOutliers__(data, m):
	matching = set();
	for i in range(len(data)):
		if (abs(data[i] - np.mean(data)) < m * np.std(data)):
			matching.add(i)
	return matching

def removeOutliers(data, normalizationParams):
	sets = list()
	for i in range(len(data)):
		sets.append(_rejectOutliers__(data[i], normalizationParams[i]))
		
	match = set.intersection(*sets)
	filteredData = list()
	
	for d in data:
		filteredData.append([d[i] for i in match])
	
	return filteredData


def generateScaler(data):

	filteredData = np.array(list(zip(*data)))
	scaler = preprocessing.StandardScaler().fit(filteredData)
	#scaledData = scaler.transform(filteredData)
	
	return scaler

def scaleData(data, scaler):
	scaledData = scaler.transform(np.array(list(zip(*data))))
	return np.array(list(zip(*scaledData)))
	
def createRegression(data, expected, regressor):
	reg2 = deepcopy(regressor)
	reg2.fit(np.array(list(zip(*data))), expected)
	return reg2

def predict(data, regresor):
	pred = regresor.predict(np.array(list(zip(*data))))
	return pred