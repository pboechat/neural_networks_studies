import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from keras.models import load_model, Sequential
from keras.layers import Input, Dense
from utilities.data_processing import replace_zeros_with_nans, fill_nans_with_mean, scale, split_train_test
from utilities.networking import download_file
from utilities.vizualisation import plot_columns_by_class, plot_confusion_matrix

_COLUMNS_WITH_MISSING_DATA = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
_CLASSIFIER_COLUMN = 'Outcome'
_CLASS_VALUES = [0, 1]
_CLASS_LABELS = ['No Diabetes', 'Diabetes']
_EPOCHS = 200
_MODEL_FILE = 'puma_indians_diabetes_model.keras'
_DATASET_URL = 'https://onedrive.live.com/download?resid=E3637CE709BADFAF%2176635&authkey=!AJuaz58jVaVVugg'
_DATASET_FILE = 'puma_indians_diabetes_dataset.csv'


def main():
    if not os.path.exists(_DATASET_FILE):
        print(f'Downloading {_DATASET_FILE}...')
        download_file(_DATASET_URL, _DATASET_FILE)

    df = pd.read_csv(_DATASET_FILE)

    plot_columns_by_class(df,
                          classifier_column=_CLASSIFIER_COLUMN,
                          class_values=_CLASS_VALUES,
                          class_labels=_CLASS_LABELS)

    replace_zeros_with_nans(df, columns=_COLUMNS_WITH_MISSING_DATA)
    fill_nans_with_mean(df, columns=_COLUMNS_WITH_MISSING_DATA)

    df = scale(df, preserve_columns=[_CLASSIFIER_COLUMN])

    X_train, X_test, y_train, y_test = split_train_test(df, classifier_column=_CLASSIFIER_COLUMN)
    X_train, X_val, y_train, y_val = split_train_test(df, classifier_column=_CLASSIFIER_COLUMN)

    if not os.path.exists(_MODEL_FILE):
        model = Sequential()

        # very simple MLP binary classifier
        model.add(Input(shape=(8,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=_EPOCHS)

        model.save(_MODEL_FILE)
    else:
        print(f'Loading {_MODEL_FILE}...')
        model = load_model(_MODEL_FILE)

    scores = model.evaluate(X_train, y_train)
    print(f'Training accuracy: {scores[1] * 100:.2f}%')

    scores = model.evaluate(X_test, y_test)
    print(f'Testing accuracy: {scores[1] * 100:.2f}%')

    # silly replacement for model.predict_classes()
    y_pred = (model.predict(X_test) > 0.5).astype('int8').flatten()
    plot_confusion_matrix(y_test, y_pred, class_labels=_CLASS_LABELS)


if __name__ == '__main__':
    main()
