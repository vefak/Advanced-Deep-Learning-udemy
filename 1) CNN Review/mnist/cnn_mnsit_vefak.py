# %% libraries
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense,BatchNormalization
from keras.utils import to_categorical
from keras.datasets.mnist import load_data
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")



# %% load and preprocess
def load_and_preprocess(data_path):
    
    data = pd.read_csv(data_path)
    data = data.values
    #np.random.shuffle(data)
    x = data[:,1:].reshape(-1,28,28,1)/255.0
    y = data[:,0].astype(np.int32)
    y = to_categorical(y, num_classes=len(set(y)))

    return x,y


train_data_path = "/home/makman4/Documents/data/mnsit_csv/mnist_train.csv"
test_data_path = "/home/makman4/Documents/data/mnsit_csv/mnist_test.csv"


X_train, Y_train = load_and_preprocess(train_data_path)
x_test, y_test = load_and_preprocess(test_data_path)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=None,shuffle =True)
print("x_train shape",X_train.shape)
print("x_val shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_val shape",Y_val.shape)

# %% visualize
index = 0
plt.imshow(X_train[index,:,:]) 
plt.legend()
plt.axis("off")
plt.show()


# %% Params
classNumber = Y_train.shape[1]
input_shape = X_train[0].shape
batch_size = 64

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
#%% CNN Model

model = Sequential()
model.add(Conv2D(16,(3,3),input_shape = input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())


model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())


model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Dense(classNumber)) # output
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = optimizer,
              metrics = ["accuracy"])


# %% Train
hist = model.fit( X_train, Y_train, validation_data=(X_val, Y_val), epochs= 25, batch_size = batch_size)
#%%
model.save_weights('cnn_mnist_model_vefak.h5')  # always save your weights after training or during training
#%% evaluation 
print(hist.history.keys())
plt.plot(hist.history["loss"],label = "Train Loss")
plt.plot(hist.history["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"],label = "Train Accuracy")
plt.plot(hist.history["val_accuracy"],label = "Validation Accuracy")
plt.legend()
plt.show()

#%% save history
import json
with open('cnn_mnist_hist.json', 'w') as f:
    json.dump(hist.history, f)
    
# #%% load history
# import codecs
# with codecs.open("cnn_mnist_hist.json", 'r', encoding='utf-8') as f:
#     h = json.loads(f.read())

# plt.figure()
# plt.plot(h["loss"],label = "Train Loss")
# plt.plot(h["val_loss"],label = "Validation Loss")
# plt.legend()
# plt.show()
# plt.figure()
# plt.plot(h["acc"],label = "Train Accuracy")
# plt.plot(h["val_acc"],label = "Validation Accuracy")
# plt.legend()
# plt.show()



# %% Eveluate

# confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict the values from the test dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# %% Score

score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

















