import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense
from diabetes_data_processing import replace_zeros_with_nans, fill_nans_with_mean, scale, split_train_test
from diabetes_data_vizualization import plot_columns_by_class, plot_confusion_matrix

_COLUMNS_WITH_MISSING_DATA = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
_CLASSIFIER_COLUMN = 'Outcome'
_CLASS_VALUES = [0, 1]
_CLASS_LABELS = ['No Diabetes', 'Diabetes']
_EPOCHS = 200


def main():
    df = pd.read_csv('diabetes.csv')

    plot_columns_by_class(df,
                          classifier_column=_CLASSIFIER_COLUMN,
                          class_values=_CLASS_VALUES,
                          class_labels=_CLASS_LABELS)

    replace_zeros_with_nans(df, columns=_COLUMNS_WITH_MISSING_DATA)
    fill_nans_with_mean(df, columns=_COLUMNS_WITH_MISSING_DATA)

    df = scale(df, preserve_columns=[_CLASSIFIER_COLUMN])

    X_train, X_test, y_train, y_test = split_train_test(df, classifier_column=_CLASSIFIER_COLUMN)
    X_train, X_val, y_train, y_val = split_train_test(df, classifier_column=_CLASSIFIER_COLUMN)

    model = Sequential()

    model.add(Input(shape=(8,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=_EPOCHS)

    scores = model.evaluate(X_train, y_train)
    print(f'Training accuracy: {scores[1] * 100:.2f}%')

    scores = model.evaluate(X_test, y_test)
    print(f'Testing accuracy: {scores[1] * 100:.2f}%')

    # silly replacement for model.predict_classes()
    y_pred = (model.predict(X_test) > 0.5).astype('int8').flatten()
    plot_confusion_matrix(y_test, y_pred, class_labels=_CLASS_LABELS)


if __name__ == '__main__':
    main()
