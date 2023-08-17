from urllib.request import urlopen
from zipfile import ZipFile
import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder # install with scikit-learn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
import visualkeras as vk # tool to help visualize the nn

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

# prepare target data as one-hot encoded
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
train_data, validation_data, train_labels, validation_labels = train_test_split(xData, yDataOH, test_size=0.2,random_state=42)

# # write the data to a file
# np.savetxt('train_data.csv', train_data, delimiter=',')
# np.savetxt('validation_data.csv', validation_data, delimiter=',')
# np.savetxt('train_labels.csv', train_labels, delimiter=',')
# np.savetxt('validation_labels.csv', validation_labels, delimiter=',')

# Build the Model
print('building model')
numInputFeatures = len(xData[0])
numExamples = len(xData)
numOutputs = len(yDataOH[0])

model = Sequential()
# add layers
model.add(Dense(numInputFeatures, input_dim=numInputFeatures, activation = 'relu', kernel_initializer= 'he_normal'))
model.add(BatchNormalization())
model.add(Dense(100,activation = 'relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(numOutputs, activation='sigmoid'))

# tensorboard_callback = TensorBoard(log_dir="./log") #tensorboard visuals

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary() # display details of model

print('preprocessing complete - beginning to train model')
#history = model.fit(train_data, train_labels, epochs=100, batch_size = 500, verbose=0, validation_data=(validation_data, validation_labels)) # train - 100 epochs, batch_size = 500
history_ret = model.fit(train_data, train_labels, epochs=5, batch_size = 500, verbose=0) # train - 100 epochs, batch_size = 500
print('training complete')

#save the model with iterative filename
save_model(model, f'sigkdd_{file_save_val}.model')


print(history_ret.history.keys())

#model accuracy
_,accuracy = model.evaluate(validation_data, validation_labels, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

# #summarize history for accuracy
# plt.plot(history_ret.history['accuracy'])
# plt.plot(history_ret.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# #summarize history for loss
# plt.plot(history_ret.history['loss'])
# plt.plot(history_ret.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlael('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # Graph Accuracy
# print('graphing accuracy')
# acc = np.asarray(history_ret.history['accuracy'])
# val_acc = np.asarray(history_ret.history['val_accuracy'])
# loss = np.asarray(history_ret.history['loss']) 
# val_loss = np.asarray(history_ret.history['val_loss'])
# epochs = np.asarray(range(len(acc)))
# plt.plot(epochs, acc, label='Training Accuracy')
# plt.plot(epochs, val_acc, label='Validation Accuracy')
# plt.plot(epochs, loss, label='Loss')
# # plt.plot(epochs, val_loss, label='Validation Loss')
# plt.title('Training/Validation Accuracy and Loss') 
# plt.xlabel('Epochs') 
# plt.ylabel('Accuracy and Loss') 
# plt.legend() 
# plt.show()
# plt.savefig(f'mode_visual_{file_save_val}.png')

# graph roc curve
print('graphing roc curve')
predictions = []
count = 0
for i in validation_data:
    #break once i is 1000
    if count == 1000:
        break
    pred = model.predict(np.array([i,]), verbose=0)
    pred = pred[0][0]
    predictions.append(pred)
    if count % 1000 == 0:
        print(f'prediction {count} complete')
    count += 1
print('predictions complete')
# calculate the roc curve for the model
# fpr, tpr, thresholds = roc_curve(validation_labels, predictions)
# calculate the roc curve for the model using only the first 1000 predictions and validation labels
roc_validation_labels = np.array(validation_labels[:1000])
predictions = np.array(predictions)

# write prediction and validation labels to the same file
np.savetxt(f'predictions_validation_labels.csv', np.column_stack((predictions, roc_validation_labels)), delimiter=',', header='predictions, validation_labels')

fpr, tpr, thresholds = roc_curve(roc_validation_labels, predictions)
# calculate tthe area under the roc curve
auc_score = roc_auc_score(roc_validation_labels, predictions)
print('AUC: %.2f' % auc_score)

# plot the roc curve for the model
plt.title = 'Receiver Operating Characteristic (ROC) Curve)'
plt.subtitle = 'Area Under Curve (AUC): %.2f' % auc_score
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.axis([0,1,0,1])
#plt.plot([0,1], [0,1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.legend()
plt.show()
plt.savefig(f'roc_curve_{file_save_val}.png')

#visualize completed neural network
# vk.layered_view(model).show()

#complexity or inputs
