import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging

from keras import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error

from utilities.data_processing import split_train_test, scale

from utilities.base_predictor import BasePredictor, run_predictor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class NycTaxiFarePredictor(BasePredictor):
    DATASET_URL = 'https://onedrive.live.com/download?resid=E3637CE709BADFAF%2176633&authkey=!AJXUPvwo49cXlhc'
    DATASET_LOAD_KWARGS = {
        'parse_dates': ['pickup_datetime'],
        'nrows': 100000
    }
    _NYC_LON_INTERVAL = [-74.05, -73.75]
    _NYC_LAT_INTERVAL = [40.63, 40.85]
    _NYC_AIRPORTS = [
        ('JFK_Airport', -73.78, 40.643),
        ('Laguardia_Airport', -73.87, 40.77),
        ('Newark_Airport', -74.18, 40.69),
    ]
    _NYC_LANDMARKS = [
                         ('Midtown', -73.98, 40.76),
                         ('Lower_Manhattan', -74.00, 40.72),
                         ('Upper_Manhattan', -73.94, 40.82),
                         ('Brooklyn', -73.95, 40.66),
                     ] + _NYC_AIRPORTS
    _MAX_FARE_AMOUNT = 100.0
    _TEMPORAL_FEATURES = ['hour', 'day', 'month', 'year', 'dayofweek']
    _EPOCHS = 1

    def _plot_map(self, lon_list, lat_list, title):
        plt.figure(figsize=(12, 12))
        plt.plot(lon_list, lat_list, '.', markersize=1)

        for landmark in self._NYC_LANDMARKS:
            plt.plot(landmark[1], landmark[2], '*', markersize=15, alpha=1, color='r')
            plt.annotate(landmark[0], (landmark[1] + 0.005, landmark[2] + 0.005), color='r', backgroundcolor='w')

        plt.grid(None)
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def _filter_lat_lon_outliers(self, df):
        for lon_col in ['pickup_longitude', 'dropoff_longitude']:
            df = df[(df[lon_col] > self._NYC_LON_INTERVAL[0]) & (df[lon_col] < self._NYC_LON_INTERVAL[1])]

        for lat_col in ['pickup_latitude', 'dropoff_latitude']:
            df = df[(df[lat_col] > self._NYC_LAT_INTERVAL[0]) & (df[lat_col] < self._NYC_LAT_INTERVAL[1])]

        return df

    def visualize_data(self, df):
        df = self._filter_lat_lon_outliers(df)

        self._plot_map(list(df.pickup_longitude), list(df.pickup_latitude), 'Pick-up')
        self._plot_map(list(df.dropoff_longitude), list(df.dropoff_latitude), 'Drop-off')

        # plot day-of-week histogram
        df.pickup_datetime.dt.dayofweek.plot.hist(bins=np.arange(8) - 0.5, ec='black')
        plt.xlabel('(0=Monday, 6=Sunday)')
        plt.title('Day-of-Week Histogram')
        plt.show()

        # plot hour histogram
        df.pickup_datetime.dt.hour.plot.hist(bins=24, ec='black')
        plt.xlabel('Hour')
        plt.title('Hour Histogram')
        plt.show()

    def create_model(self, df):
        # since we have enough data for training/testing, drop nans
        df = df.dropna()

        # filter fare amount outliers
        df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < self._MAX_FARE_AMOUNT)]

        # replace invalid passenger counts with mean
        df.loc[df['passenger_count'] == 0, 'passenger_count'] = df['passenger_count'].mean()

        df = self._filter_lat_lon_outliers(df)

        # create temporal features
        for attr in self._TEMPORAL_FEATURES:
            df[attr] = getattr(df.pickup_datetime.dt, attr)

        df = df.drop(['pickup_datetime'], axis=1)

        def distance(lat_1, lon_1, lat_2, lon_2):
            return ((lat_1 - lat_2) ** 2 + (lon_1 - lon_2) ** 2) ** 0.5

        df['distance'] = distance(df.pickup_latitude, df.pickup_longitude,
                                  df.dropoff_latitude, df.dropoff_longitude)

        for airport in self._NYC_AIRPORTS:
            df['pickup_distance_' + airport[0]] = distance(df.pickup_latitude, df.pickup_longitude,
                                                           airport[1], airport[2])
            df['dropoff_distance_' + airport[0]] = distance(df.dropoff_latitude, df.dropoff_longitude,
                                                            airport[1], airport[2])

        df = df.drop(['key'], axis=1)

        df = scale(df, preserve_columns=['fare_amount'], ignore_columns=['fare_amount'])

        X_train, X_test, y_train, y_test = split_train_test(df, target_column='fare_amount')

        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1))

        model.summary()

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mse'])

        model.fit(X_train, y_train, epochs=self._EPOCHS)

        train_mse = mean_squared_error(y_train, self.predict(model, X_train))
        test_mse = mean_squared_error(y_test, self.predict(model, X_test))

        logging.info(f'Train MSE: {train_mse:.2f}')
        logging.info(f'Test MSE: {test_mse:.2f}')

    def predict(self, model, df):
        return model.predict(df)


if __name__ == '__main__':
    run_predictor()
