import math

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

_LINE_STYLES = ['-', '--', '-.', ':']


def plot_histogram(df):
    df.hist()
    plt.show()


def plot_columns_by_class(df, *, classifier_column, class_values=(0, 1), class_labels=('Zero', 'One')):
    side = int(math.ceil(math.sqrt(len(df.columns) - 1)))
    plt.subplots(side, side, figsize=(15, 15))
    i = 0
    for column in df.columns:
        if column == classifier_column:
            continue
        i += 1
        ax = plt.subplot(side, side, i)
        ax.yaxis.set_ticklabels([])
        for j, clazz in enumerate(class_labels):
            sns.distplot(df.loc[df[classifier_column] == class_values[j]][column],
                         hist=False,
                         axlabel=False,
                         kde_kws={
                             'linestyle': _LINE_STYLES[j % len(_LINE_STYLES)],
                             'color': 'black',
                             'label': class_labels[j]
                         })
        ax.set_title(column)
    for j in range(len(df.columns), (side * side) + 1):
        plt.subplot(side, side, j).set_visible(False)
    plt.show()


def plot_confusion_matrix(y_test, y_pred, *, class_labels=('Zero', 'One')):
    ax = sns.heatmap(confusion_matrix(y_test, y_pred),
                     xticklabels=class_labels,
                     yticklabels=class_labels,
                     annot=True,
                     cbar=False)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Actual')
    plt.show()
