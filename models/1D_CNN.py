#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau


# In[2]:


data = pd.read_csv("../data.csv")
data.head()


# In[3]:


data.drop("Unnamed: 0.1", axis = 1, inplace = True)
data.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
data.replace(0, np.nan, inplace=True)


# In[4]:


data


# In[5]:


data.interpolate(method='linear', limit_direction='forward')


# In[6]:


data.interpolate(method='linear', limit_direction='forward', inplace=True)
LOOK_BACK = 30
PREDICT_DAY = 1
SPLIT_RATIO = 0.8


# In[7]:


def Create_Data(
    data, lookback=LOOK_BACK, pred_len=PREDICT_DAY, split_ratio=SPLIT_RATIO
):
    if lookback < 2:
        print("ERROR: Lookback too small")
        return -1

    # declarations

    x = {}
    y = {}
    xtr = {}
    xt = {}
    ytr = {}
    yt = {}
    scalers = {}

    # Creating stepped data

    for i in data.columns:
        xtemp = pd.DataFrame(data[i])
        for j in range(1, lookback + 1):
            xtemp[i + str(j)] = data[i].shift(-1 * j)
        x[i] = xtemp.dropna()

    # Splitting data into x and y

    for i in x.keys():
        y[i] = pd.DataFrame(x[i].iloc[:, -pred_len])
        x[i] = x[i].iloc[:, :-pred_len]

    # Normalizing x and y values

    for i in x.keys():
        scalers[i + "_x"] = MinMaxScaler(feature_range=(0, 1))
        x[i] = scalers[i + "_x"].fit_transform(x[i])
        scalers[i + "_y"] = MinMaxScaler(feature_range=(0, 1))
        y[i] = scalers[i + "_y"].fit_transform(y[i])

    # setting train and test sizes

    tr_len = int(split_ratio * y["India"].shape[0])
    t_len = y["India"].shape[0] - tr_len

    # creating training and testing data

    for i in x.keys():
        xtr[i] = x[i][:tr_len]
        ytr[i] = y[i][:tr_len]
        xt[i] = x[i][-t_len:]
        yt[i] = y[i][-t_len:]

    # returning pertinent data

    return x, y, xtr, xt, ytr, yt, scalers


# In[8]:


x,y,xtr,xt,ytr,yt,scalers = Create_Data(data)


# In[9]:


def Create_model(x, lookback=LOOK_BACK):
    models = {}
    for key in x.keys():
        input_dim = x[key].shape[
            1
        ]  # Assuming x[key] is a 2D numpy array, where the second dimension is the feature size

        model = Sequential()

        # Convolutional Layers
        model.add(Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(input_dim, 1)))
        model.add(Conv1D(filters=64, kernel_size=7, activation='relu'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
        model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
        model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())

        # Dense Layers
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))

        # Output Layer
        model.add(Dense(1, activation="linear"))

        model.compile(optimizer="adam", loss="mean_squared_error")

        models[key] = model

    return models


# In[10]:


m = Create_model(x,y)


# In[11]:


for key in xtr:
    print(f"Shape of xtr[{key}] = {xtr[key].shape}")
    print(f"Shape of xt[{key}] = {xt[key].shape}")


# In[12]:


def Execute_model(model, xtr, ytr, xt, yt, scaler):
    MAPE = {}
    MAE = {}
    MSE = {}
    for i in model.keys():
        print(i)
        # Creating EarlyStopping and ReduceLROnPlateau callbacks
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
        )

        # Reshaping data for Conv1D
        xtr_reshaped = np.reshape(xtr[i], (xtr[i].shape[0], xtr[i].shape[1], 1))
        xt_reshaped = np.reshape(xt[i], (xt[i].shape[0], xt[i].shape[1], 1))

        # Training the model with EarlyStopping and ReduceLROnPlateau callbacks
        model[i].fit(
            xtr_reshaped,
            ytr[i],
            epochs=100,
            batch_size=64,
            verbose=1,
            validation_split=0.2,
            callbacks=[es, reduce_lr],
        )

        # collecting predicted and actual values
        temp = model[i].predict(xt_reshaped)
        pred = scaler[i + "_y"].inverse_transform(temp)
        act = scaler[i + "_y"].inverse_transform(yt[i])

        # calculating Mean Square Error, Mean Absolute Error, and Mean Absolute Error
        MSE[i] = mean_squared_error(act, pred)
        MAE[i] = mean_absolute_error(act, pred)
        MAPE[i] = mean_absolute_percentage_error(act, pred)

    # Tabulating Data
    results = pd.DataFrame([MSE, MAE, MAPE])
    results["Metric"] = ["MSE", "MAE", "MAPE"]
    results.set_index("Metric", inplace=True)

    return results


# In[13]:


result = Execute_model(m,xtr,ytr,xt,yt,scalers)


# In[14]:


result


# In[15]:


import json
import os

# Define the filename
filename = "../results.json"

# Check if the file exists
if os.path.isfile(filename):
    # If the file exists, load the existing data
    with open(filename, "r") as f:
        data = json.load(f)
else:
    # If the file doesn't exist, create an empty dictionary
    data = {}

# Add the result to the dictionary
data["1D_CNN"] = result.to_dict()

# Write the dictionary to the file
with open(filename, "w") as f:
    json.dump(data, f)

