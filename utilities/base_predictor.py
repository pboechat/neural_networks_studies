import os
import re
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser
import logging.config

import pandas as pd
from keras.models import load_model

from utilities.networking import download_file


def camel_to_snake(name):
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    return name


def load_dataframe(filename, *, url=None, **kwargs):
    if not filename:
        raise Exception('invalid dataframe')

    if not os.path.exists(filename):
        if url is not None:
            logging.info(f'downloading {filename}...')
            download_file(url, filename)

    return pd.read_csv(filename, **kwargs)


class BasePredictor(ABC):
    DATASET_URL = None
    DATASET_EXT = '.csv'
    DATASET_LOAD_KWARGS = {}
    MODEL_EXT = '.keras'

    def __init__(self):
        super().__init__()

    @property
    def dataset_name(self):
        return f'{camel_to_snake(self.__class__.__name__).replace("_predictor", "") + "_dataset"}{self.DATASET_EXT}'

    @property
    def model_name(self):
        return f'{camel_to_snake(self.__class__.__name__).replace("_predictor", "") + "_model"}{self.MODEL_EXT}'

    def _load_dataframe(self):
        return load_dataframe(self.dataset_name, url=self.DATASET_URL, **self.DATASET_LOAD_KWARGS)

    def _visualize_data(self):
        df = self._load_dataframe()
        self.visualize_data(df)

    def visualize_data(self, df):
        pass

    def _create_model(self):
        df = self._load_dataframe()
        model = self.create_model(df)
        if model is None:
            raise Exception('invalid model')
        model.save(self.model_name)
        return model

    @abstractmethod
    def create_model(self, df):
        pass

    def _predict(self, df):
        model_name = self.model_name
        if os.path.exists(model_name):
            model = load_model(model_name)
        else:
            model = self._create_model()
        y_pred = self.predict(model, df)

        # TODO:
        logging.info(y_pred)

        return y_pred

    @abstractmethod
    def predict(self, model, df):
        pass


def run_predictor():
    parser = ArgumentParser()
    parser.add_argument('--mode', '-m', type=str,
                        choices=['visualize_data', 'create_model', 'predict'],
                        required=False, default='predict')
    parser.add_argument('--dataframe', '-df', type=str, required=False)
    args = parser.parse_args()

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '[%(asctime)s] - %(levelname)s - %(message)s',
                'datefmt': '%d-%m-%Y %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': 'log.txt',
                'mode': 'w'
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
    })

    subclasses = BasePredictor.__subclasses__()

    if not subclasses:
        logging.error(f'no {BasePredictor.__name__} subclass found')
        sys.exit(-1)

    subclass_instance = subclasses[0]()

    if args.mode == 'visualize_data':
        subclass_instance._visualize_data()
    elif args.mode == 'create_model':
        subclass_instance._create_model()
    elif args.mode == 'predict':
        if not args.dataframe:
            logging.error('missing dataframe')
            sys.exit(-1)
        subclass_instance._predict(load_dataframe(args.dataframe))
    else:
        # checking invariants
        logging.error(f'invalid mode: {args.mode}')
        sys.exit(-1)

    sys.exit(0)
