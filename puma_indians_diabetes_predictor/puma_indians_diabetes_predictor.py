import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import Sequential
from keras.layers import Input, Dense
from utilities.base_predictor import BasePredictor, run_predictor
from utilities.data_processing import replace_zeros_with_nans, fill_nans_with_mean, scale, split_train_test
from utilities.vizualisation import plot_columns_by_class, plot_confusion_matrix
import logging


class PumaIndiansDiabetesPredictor(BasePredictor):
    DATASET_URL = 'https://onedrive.live.com/download?resid=E3637CE709BADFAF%2176635&authkey=!AJuaz58jVaVVugg'
    _COLUMNS_WITH_MISSING_DATA = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    _CLASS_VALUES = [0, 1]
    _CLASS_LABELS = ['No Diabetes', 'Diabetes']
    _EPOCHS = 200

    def visualize_data(self, df):
        plot_columns_by_class(df,
                              target_column='Outcome',
                              class_values=self._CLASS_VALUES,
                              class_labels=self._CLASS_LABELS)

    def create_model(self, df):
        replace_zeros_with_nans(df, columns=self._COLUMNS_WITH_MISSING_DATA)
        fill_nans_with_mean(df, columns=self._COLUMNS_WITH_MISSING_DATA)

        df = scale(df, preserve_columns=['Outcome'])

        X_train, X_test, y_train, y_test = split_train_test(df, target_column='Outcome')
        X_train, X_val, y_train, y_val = split_train_test(df, target_column='Outcome')

        model = Sequential()

        # very simple MLP binary classifier
        model.add(Input(shape=(8,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=self._EPOCHS)

        scores = model.evaluate(X_train, y_train)
        logging.info(f'Training accuracy: {scores[1] * 100:.2f}%')

        scores = model.evaluate(X_test, y_test)
        logging.info(f'Testing accuracy: {scores[1] * 100:.2f}%')

        y_pred = self.predict(model, X_test)

        plot_confusion_matrix(y_test, y_pred, class_labels=self._CLASS_LABELS)

        return model

    def predict(self, model, df):
        # silly replacement for model.predict_classes()
        return (model.predict(df) > 0.5).astype('int8').flatten()


if __name__ == '__main__':
    run_predictor()
