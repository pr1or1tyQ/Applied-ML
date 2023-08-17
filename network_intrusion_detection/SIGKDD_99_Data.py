import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from urllib.request import urlopen
from zipfile import ZipFile
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, roc_auc_score # roc curve tools
import matplotlib.pyplot as plt 

#Download the file from KDD Website. Comment this out after you have downloaded the file once.
zipurl = 'http://kdd.org/cupfiles/KDDCupData/1999/kddcup.data.zip'
zipresp = urlopen(zipurl)

tempzip = open('kddcup.zip', 'wb')
tempzip.write(zipresp.read())
tempzip.close()

zf = ZipFile("kddcup.zip")
zf.extractall()

#End comment here if you have already downloaded the zip file

file = open("kddcup.data.txt")
fullDataList = list(csv.reader(file, delimiter=","))
file.close()

#One row of data has 56 columns instead of 42 for some reason. Deleting this row. 
for item in fullDataList:
    if len(item) > 42:
        fullDataList.remove(item)

npArray = np.array(fullDataList)

#Split into input and output data
xData = npArray[:,:-1]
yData = npArray[:,-1]

#prepare the target data as one-hot encoded
one = OneHotEncoder(dtype=np.float64)
yData = np.reshape(yData, (-1, 1))
one.fit(yData)
yDataOH = one.transform(yData).toarray() 

#prepare the input data so that strings are encoded as integers
oe = OrdinalEncoder(dtype=np.float64)
oe.fit(xData[:,1:4])
xDataOE = oe.transform(xData[:,1:4])
xData[:,1:4] = xDataOE
xData = xData.astype(np.float64)

#Now let's divide the data into testing and trainnig and try to run on a Neural Network
XTrain, XTest, YTrain, YTest = train_test_split(xData, yDataOH, test_size = 0.2, random_state = 42)

#Model Build
numInputFeatures = len(xData[0])
numExamples = len(xData)
numOutputs = len(yDataOH[0])

model = Sequential()
model.add(Dense(numInputFeatures, input_dim=numInputFeatures, activation = 'relu', kernel_initializer = 'he_normal'))
model.add(BatchNormalization())
model.add(Dense(100, activation = 'relu')) 
model.add(Dense(50, activation = 'relu'))
model.add(Dense(numOutputs, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(XTrain, YTrain, epochs = 100, batch_size = 500)

model.summary()
_,accuracy = model.evaluate(XTest, YTest, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

# compare predictions to actual labels
# write the misclassified data to 'misclcassified_data.txt'
file = open('misclassified_data.txt', 'w')

predictions = []
count = 0
for i in XTest:
    #break once i is 1000
    # if count == 1000:
    #     print(len(i))
    #     print(pred)
    #     break
    pred = model.predict(np.array([i,]), verbose=0)
    predictions.append(pred[0])
    if count % 1000 == 0:
        print(f'prediction {count} complete')
    count += 1
print('predictions complete')
print(len(predictions))

#compare predictions to actual labels, record misclassified ones
for i in range (len(predictions)):
    if predictions[i].argmax() == YTest[i].argmax():
        continue
    else:
        # write the misclassified data to a file
        file.write(f'Missclassified data: {predictions[i]}\nActual label: {YTest[i]}')



# show if prediciton is false positive
# show if prediction is false negative
# show if prediction is true positive
# show if prediction is true negative



