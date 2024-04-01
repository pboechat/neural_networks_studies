import os

from utilities.networking import download_file

_TEST_DATASET_URL = 'https://onedrive.live.com/download?resid=E3637CE709BADFAF%2176632&authkey=!ACEYtuoF_7h7cig'
_TEST_DATASET_FILE = 'ny_taxi_fare_test_dataset.csv'
_TRAIN_DATASET_URL = 'https://onedrive.live.com/download?resid=E3637CE709BADFAF%2176633&authkey=!AJXUPvwo49cXlhc'
_TRAIN_DATASET_FILE = 'ny_taxi_fare_train_dataset.csv'


def main():
    if not os.path.exists(_TEST_DATASET_FILE):
        print(f'Downloading {_TEST_DATASET_FILE}...')
        download_file(_TEST_DATASET_URL, _TEST_DATASET_FILE)

    if not os.path.exists(_TRAIN_DATASET_FILE):
        print(f'Downloading {_TRAIN_DATASET_FILE}...')
        download_file(_TRAIN_DATASET_URL, _TRAIN_DATASET_FILE)

    # TODO:
    pass


if __name__ == '__main__':
    main()
