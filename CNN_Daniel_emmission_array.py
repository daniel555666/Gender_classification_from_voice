import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
#import seaborn as sns
import os
import random
import cv2
import json
from sklearn.model_selection import train_test_split
from torch import nn
import constants as c
import torch


NUM_OF_SPEAKERS = 2
DIMENSION = 512 * 300





np.random.seed(0)


# Load data into dataframe
training_data_folder = 'data\json_russian_data'
data = []

for folder in sorted(os.listdir(training_data_folder)):
    if folder[0] == '.':
        continue
    sub_folder = os.path.join(training_data_folder,folder)
    files = [{'label':folder,'path':os.path.join(sub_folder, f)} for f in os.listdir(sub_folder) if os.path.isfile(os.path.join(sub_folder, f))]
    data += files

df = pd.DataFrame(data)

map_characters = {0: 'M', 1: 'W'}

order_list = ['M', 'W']

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
hotEncodedLabels = lb.fit_transform(order_list)
hotEncodedLabels

#This code will only load the folders for 'M' (man) and 'W' (woman) from the training data folder, and label the data as either "man" or "woman".
#The hot encoded labels will also be changed to [1,0] for "man" and [0,1] for "woman".
train_dir = 'data\json_russian_data'

map_characters = {0: 'M', 1: 'W'}

labels_dict = {'M': 0, 'W': 1}

order_list = ('M', 'W')

def load_data():
    """
    Loads data and preprocess. Returns train and test data along with labels.
    """
    images = []
    labels = []
    size = 29,400
    num=0
    count=0

    print("LOADING DATA FROM : ",end = "")
    for folder in os.listdir(train_dir): ## to check 
        if folder[0] == '.':
            continue
        print(folder, end = ' | ')
        for json_file in os.listdir(train_dir + "/" + folder):
            temp_json_file = json.load(open(train_dir + '/' + folder + '/' + json_file))
            temp_np_array = np.array(temp_json_file)
            temp = [cv2.resize(temp_np_array[0], size)]
            temp_img = np.array(temp)
            # plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
            # plt.show()
            images.append(temp_img)
            labels.append(num)
            count+=1 #count the number of photos
            if count == 1000:
                print("data number: ",count ,"was load")
        num+=1
    
    
    images = np.array(images)
    # images = images.astype('float32')/255
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.3, random_state=42)
    X_test, X_validation, Y_test, Y_validation = train_test_split(X_test, Y_test, test_size = 0.8,random_state=42)
    
    
    print()
    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)
    print('Loaded', len(X_validation),'images for validation','validation data shape =',X_validation.shape)
    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)
    print('\n')

    
    return X_train, X_test, Y_train, Y_test, X_validation, Y_validation

X_train, X_test, Y_train, Y_test, X_validation, Y_validation= load_data()


DROP_OUT = 0.5

class Convolutional_Speaker_Identification(nn.Module):

    def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
        return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2

    def __init__(self):

        super().__init__()

        self.conv_2d_1 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=2)
        self.bn_1 = nn.BatchNorm2d(96)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2d_2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.bn_2 = nn.BatchNorm2d(256)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2d_3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=2)
        self.bn_3 = nn.BatchNorm2d(384)

        self.conv_2d_4 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=2)
        self.bn_4 = nn.BatchNorm2d(256)

        self.conv_2d_5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=2)
        self.bn_5 = nn.BatchNorm2d(256)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))

        self.conv_2d_6 = nn.Conv2d(256, 4096, kernel_size=(9, 1), padding=0)
        self.drop_1 = nn.Dropout(p=DROP_OUT)

        self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_1 = nn.Linear(4096, 1024)
        self.drop_2 = nn.Dropout(p=DROP_OUT)

        self.dense_2 = nn.Linear(1024, NUM_OF_SPEAKERS)

    def forward(self, X):

        x = nn.ReLU()(self.conv_2d_1(X))
        x = self.bn_1(x)
        x = self.max_pool_2d_1(x)

        x = nn.ReLU()(self.conv_2d_2(x))
        x = self.bn_2(x)
        x = self.max_pool_2d_2(x)

        x = nn.ReLU()(self.conv_2d_3(x))
        x = self.bn_3(x)

        x = nn.ReLU()(self.conv_2d_4(x))
        x = self.bn_4(x)

        x = nn.ReLU()(self.conv_2d_5(x))
        x = self.bn_5(x)
        x = self.max_pool_2d_3(x)

        x = nn.ReLU()(self.conv_2d_6(x))
        x = self.drop_1(x)
        x = self.global_avg_pooling_2d(x)

        x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer
        x = nn.ReLU()(self.dense_1(x))
        x = self.drop_2(x)

        x = self.dense_2(x)
        y = nn.LogSoftmax(dim=1)(x)   # consider using Log-Softmax

        return y

    def get_epochs(self):
        return 3

    def get_learning_rate(self):
        return 0.0001

    def get_batch_size(self):
        return 16

    def to_string(self):
        return "Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_"


import torch.nn as nn
import torch.optim as optim

# Define loss, optimizer and metrics
model = Convolutional_Speaker_Identification()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_fn = nn.CrossEntropyLoss()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
metrics = ["accuracy"]

import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("using cuda")
else:
    print("not using cuda")
    device = torch.device("cpu")

data_x = torch.tensor(X_train).to(device)
data_y = torch.tensor(Y_train).to(device)

val_x = torch.tensor(X_validation).to(device)
val_y = torch.tensor(Y_validation).to(device)

# training
batch_size = 10
epochs = 10

model = model.to(torch.float)


for i in range(epochs):
    for j in range(len(data_x) // batch_size):
        start_idx = j * batch_size
        end_idx = start_idx + batch_size
        inputs = data_x[start_idx:end_idx]
        inputs = inputs.to(torch.float)
        targets = data_y[start_idx:end_idx]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        model.conv_2d_1.bias = model.conv_2d_1.bias.to(inputs.dtype)
        optimizer.step()
        if j % 100 == 0:
            print(f"Epoch {i+1}, batch {j+1}, loss: {loss.item():.4f}")
    print("eval")
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        for j in range(len(val_x) // batch_size):
            start_idx = j * batch_size
            end_idx = start_idx + batch_size
            inputs = val_x[start_idx:end_idx]
            inputs = inputs.to(torch.float)

            targets = val_y[start_idx:end_idx]
            model.conv_2d_1.bias = torch.nn.Parameter(model.conv_2d_1.bias.to(inputs.dtype))
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(1) == targets).sum().item()
        accuracy = total_correct / len(val_x)
        print(f"Validation loss: {total_loss:.4f}, Validation accuracy: {accuracy:.4f}")

#optim = torch.optim.Adam(model.parameters(), lr=0.0001)


# save the model to model.h5 file
torch.save(model.state_dict(), 'model.pt')

test_x = torch.tensor(X_test)
test_y = torch.tensor(Y_test)

#print("test")

# model.eval()
# with torch.no_grad():
#     model.conv_2d_1.bias = torch.nn.Parameter(model.conv_2d_1.bias.to(inputs.dtype))
    
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     test_x = test_x.float().to(device)
#     test_y = test_y.to(device)
#     outputs = model(test_x)
#     _, predicted = torch.max(outputs, 1)
#     total = test_y.size(0)
#     correct = (predicted == test_y).sum().item()
#     accuracy = 100 * correct / total
#     print('Accuracy of the network on the test set: {:.2f}%'.format(accuracy))

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# test_x = torch.tensor(X_test).float().to(device)
# test_y = torch.tensor(Y_test).to(device)

# with torch.no_grad():
#     model.eval()
#     pred = model(test_x)
#     pred = torch.argmax(pred, axis=1)
#     acc = accuracy_score(test_y.cpu().numpy(), pred.cpu().numpy())
#     prec = precision_score(test_y.cpu().numpy(), pred.cpu().numpy(), average='weighted')
#     rec = recall_score(test_y.cpu().numpy(), pred.cpu().numpy(), average='weighted')
#     f1 = f1_score(test_y.cpu().numpy(), pred.cpu().numpy(), average='weighted')

# # Print the results
# print("Accuracy: {:.2f}".format(acc))
# print("Precision: {:.2f}".format(prec))
# print("Recall: {:.2f}".format(rec))
# print("F1-Score: {:.2f}".format(f1))


# from sklearn.metrics import confusion_matrix
# #import seaborn as sns
# import matplotlib.pyplot as plt

# def plot_confusion_matrix(y_true, y_pred, classes):
#     cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
#     plt.figure(figsize=(10, 10))
#     ax = sns.heatmap(cf_matrix, annot=True, cmap='Greens', fmt='.2f')
#     ax.set_xticklabels(classes, rotation=45, ha='right')
#     ax.set_yticklabels(classes, rotation=0)
#     plt.xlabel('Predicted label')
#     plt.ylabel('True label')
#     plt.title('Confusion matrix')

#test_classes = ["class0", "class1", "class2", ...] # list of class names
#plot_confusion_matrix(test_y, mypred, test_classes)
