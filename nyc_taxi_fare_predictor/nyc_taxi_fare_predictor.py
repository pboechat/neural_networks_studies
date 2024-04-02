import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utilities.base_predictor import BasePredictor, run_predictor
import matplotlib.pyplot as plt
import numpy as np


class NycTaxiFarePredictor(BasePredictor):
    DATASET_URL = 'https://onedrive.live.com/download?resid=E3637CE709BADFAF%2176633&authkey=!AJXUPvwo49cXlhc'
    DATASET_LOAD_KWARGS = {
        'parse_dates': ['pickup_datetime'],
        'nrows': 50000
    }
    _NYC_LON_INTERVAL = [-74.05, -73.75]
    _NYC_LAT_INTERVAL = [40.63, 40.85]
    _NYC_LANDMARKS = [
        ('JFK Airport', -73.78, 40.643),
        ('Laguardia Airport', -73.87, 40.77),
        ('Midtown', -73.98, 40.76),
        ('Lower Manhattan', -74.00, 40.72),
        ('Upper Manhattan', -73.94, 40.82),
        ('Brooklyn', -73.95, 40.66),
    ]

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

    def visualize_data(self, df):
        for lon_col in ['pickup_longitude', 'dropoff_longitude']:
            df = df[(df[lon_col] > self._NYC_LON_INTERVAL[0]) & (df[lon_col] < self._NYC_LON_INTERVAL[1])]

        for lat_col in ['pickup_latitude', 'dropoff_latitude']:
            df = df[(df[lat_col] > self._NYC_LAT_INTERVAL[0]) & (df[lat_col] < self._NYC_LAT_INTERVAL[1])]

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
        pass

    def predict(self, model, df):
        pass


if __name__ == '__main__':
    run_predictor()
