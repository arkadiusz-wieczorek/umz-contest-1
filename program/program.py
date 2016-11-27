#  -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
from sklearn import preprocessing
from sklearn.externals import joblib

#reading data
print("Loading Data...")
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


#reading coords
coordData = list()
with open('../coords.tsv','r', encoding="utf8") as file:
	tsvin = csv.reader(file, delimiter='\t')
	for row in tsvin:
		coordData.append(row)

columnsZipped = zip(*coordData)
coordColumns = list(columnsZipped);

print("Data loaded!")

def rejectOutliers(data, m):
	matching = set();
	for i in range(len(data)):
		if (abs(data[i] - np.mean(data)) < m * np.std(data)):
			matching.add(i)
	return matching

	
normalizationParams = [0.8, 2, 2, 2, 1]

print ("Removing outliers with parameters {0}...".format(normalizationParams))

match = set.intersection( 
	rejectOutliers(prices, normalizationParams[0]),
	rejectOutliers(rooms, normalizationParams[1]),
	rejectOutliers(meters, normalizationParams[2]),
	rejectOutliers(floors, normalizationParams[3]),
	rejectOutliers(pricesPerMeter, normalizationParams[4]))

pricesFiltered = [prices[i] for i in match]
roomsFiltered = [rooms[i] for i in match]
metersFiltered = [meters[i] for i in match]
floorsFiltered = [floors[i] for i in match]
pricesPerMeterFiltered = [pricesPerMeter[i] for i in match]

print ("Outliers removed! Remaining {0} out of {1} items.".format(len(match), len(prices)))

print ("Normalizing...")

scaler = preprocessing.StandardScaler().fit([roomsFiltered, metersFiltered, floorsFiltered, pricesPerMeterFiltered])

normFile = 'norm.pkl'

joblib.dump(scaler, normFile)

print ("Normalization parameters saved in file", normFile)

scaledData = scaler.transform([roomsFiltered, metersFiltered, floorsFiltered, pricesPerMeterFiltered])

roomsNormalized = scaledData[0]
metersFiltered = scaledData[1]
floorsFiltered = scaledData[2]
pricesPerMeterFiltered = scaledData[3]

print ("Normalization finished!\n")

######__NOT_FINISHED__##############

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()