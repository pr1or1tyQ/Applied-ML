from urllib.request import urlopen
from zipfile import ZipFile
import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder # install with scikit-learn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

# from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt 
from keras.models import save_model # save model
from sklearn.metrics import roc_curve, roc_auc_score # roc curve tools

file_save_val = 1700

# Import Dataset: Download file from KDD Website
print('beginning sigkdd dataset download')
zip_url = 'http://kdd.org/cupfiles/KDDCupData/1999/kddcup.data.zip'
data_zip = urlopen(zip_url)

# write requrest to zip file
temp_zip = open('kddcup.zip','wb')
temp_zip.write(data_zip.read())
temp_zip.close()

# extract contents of zip file
zip_file = ZipFile("kddcup.zip")
zip_file.extractall()
print('dataset download complete')

file = open('kddcup.data.txt')
data_list = list(csv.reader(file, delimiter=","))
file.close()

# dataset import Complete
print('data set import complete')

#standardize length of columns - remove outlier
for item in data_list:
    if(len(item) > 42):
        data_list.remove(item)

# convert to np array
npArray = np.array(data_list)

# seperate input and output data
xData = npArray[:,:-1]
yData = npArray[:,-1]

# prepare target data as one-hot encoded ***why this step
one = OneHotEncoder(dtype=np.float64)
yData = np.reshape(yData, (-1,1)) 
one.fit(yData)
yDataOH = one.transform(yData).toarray()

# make input data so that strings are encoded as integers
oe = OrdinalEncoder(dtype=np.float64)
oe.fit(xData[:,1:4])
xDataOE = oe.transform(xData[:,1:4])
xData[:,1:4] = xDataOE
xData = xData.astype(np.float64)
print('data cleaned')

#divide the data in testing and training sets
train_data, test_data, train_labels, test_labels = train_test_split(xData, yDataOH, test_size=0.2,random_state=42)

# # write the data to a file
np.savetxt('train_data.csv', train_data, delimiter=',')
np.savetxt('test.csv', test_data, delimiter=',')
np.savetxt('train_labels.csv', train_labels, delimiter=',')
np.savetxt('test.csv', test_labels, delimiter=',')