import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import numpy as np #matrix math
from sklearn.model_selection import train_test_split #to split out training and testing data 
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input
#helper class to define input shape and generate training images given image paths & steering angles
from utils import INPUT_SHAPE, batch_generator
#for command line arguments
import argparse
#for reading files
import os

def load_data():
    test_size =0.2
    data_dir = r"C:\Users\gobli\Documents\Airsim\END_TO_END\data_raw"
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'test.csv'), names=['image_dir', 'steering'])

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df['image_dir'].values
    #and our steering commands as our output data
    y = data_df['steering'].values

    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid

def build_model():
    model = Sequential()
    model.add(Input(shape=INPUT_SHAPE))
    model.add(Lambda(lambda x: x / 127.5 - 1.0))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model

def train_model(model,  X_train, X_valid, y_train, y_valid):
    save_best_only = True
    learning_rate= 1.0e-4
    data_dir= 'data_raw'
    batch_size = 64
    samples_per_epoch = 20000
    nb_epoch = 10
    checkpoint = ModelCheckpoint('model-{epoch:03d}.keras',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=save_best_only,
                                 mode='auto',
                                 )

    
    #gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-4))

    #Fits the model on data generated batch-by-batch by a Python generator.

    
    model.fit(
        batch_generator(data_dir, X_train, y_train, batch_size, True),
        steps_per_epoch=samples_per_epoch // batch_size,
        epochs=nb_epoch,
        validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False),
        validation_steps=len(X_valid) // batch_size,
        callbacks=[checkpoint],
        verbose=1
    )

def main():
    """
    Load train/validation data set and train the model
    """


    #load data
    data = load_data()
    #build model
    model = build_model()
    #train model on data, it saves as model.h5 
    train_model(model, *data)

if __name__ == '__main__':
    main()