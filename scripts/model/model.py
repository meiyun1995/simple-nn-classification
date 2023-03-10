from ast import Tuple
import os
from utils.logging import LOG
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

from scripts.data.data_pipeline import load_data, data_preprocessing

class Model:
    def __init__(self, df: pd.DataFrame):
        self.data = df
        
    def train_test_split(self):
        """Split the data intol train and test set and return a tuple of pandas dataframe.

        Args:
            df (pd.DataFrame): cleaned data

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: tuple of train and test set
        """
        # Split data into train and test set
        X = self.data.drop('booking_status', axis=1)
        y = self.data.booking_status
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
        return X_train, X_test, y_train, y_test
        
    def normalized_data(self):
        """Normalize the data and return a tuple of numpy array.

        Args:
            X_train (pd.DataFrame): train set
            X_test (pd.DataFrame): test set
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of normalized train and test set"""
        
        X_train, X_test, y_train, y_test = self.train_test_split()
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """Build a model and return a keras sequential model.


        Returns:
            tf.keras.Sequential: keras sequential model
        """
        model = Sequential()
        model.add(Dense(units=24,activation='relu'))
        model.add(Dense(units=10,activation='relu'))
        model.add(Dense(units=5,activation='relu'))
        model.add(Dense(units=1,activation='sigmoid'))
        opt = Adam(learning_rate=0.001)
        model.compile( loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


        return model
        
    def model_pipeline(self, check_model_exists: bool = True) -> np.ndarray:
        """Train the model on normalised data set and return a keras sequential model, predictions.

        Returns:
            model (tf.keras.Sequential): keras sequential model
            predictions (np.ndarray): numpy array
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.normalized_data()

        LOG.info('Checking if model exists...')
        if check_model_exists and os.path.exists('scripts/model/nn_model.h5'):
            LOG.info('Model found! Loading trained model...')
            model = load_model('scripts/model/nn_model.h5')
        else:
            LOG.info('Model not found! Training model...')
            model = self.build_model()
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
            model.fit(x=self.X_train, y=self.y_train,
                        epochs=80, validation_data=(self.X_test, self.y_test),
                        batch_size=28, callbacks=[early_stop],
                        verbose=1)

            LOG.info('Saving model...')
            model.save('scripts/model/nn_model.h5')

        LOG.info('Making predictions...')
        predict_x = model.predict(self.X_test) 
        predictions = (predict_x > 0.5).astype("int32")

        return predictions

    def check_performance(self, predictions: np.ndarray) -> None:
        """Print the classification report and confusion matrix.
        
        Args:
            predictions (np.ndarray): numpy array
            
        Returns:
            None
        """
        print(f'Classification report: \n{(classification_report(self.y_test, predictions))} \nConfusion Matrix: \n{(confusion_matrix(self.y_test, predictions))}')
