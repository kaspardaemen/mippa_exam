import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# credits: https://www.kaggle.com/benroshan/fake-news-classifier-lstm
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def check_model_performance_as1(X_test, y_test, model):
    preds = np.array(model.predict(x = X_test))

    predictions_preds = np.array([np.argmax(x) for x in preds])
    predictions_real = y_test[:,1]

    print(f'accuracy: {accuracy_score(predictions_real, predictions_preds)}')

    cm = confusion_matrix(predictions_real, predictions_preds, labels=[1, 0], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['4top', 'background'])

    disp = disp.plot()
    plt.show()

def check_model_performance_as2(X_test, y_test, model):
    preds = np.array(model.predict(x = X_test))
    #prior = np.array([0.55, 0.45])

    #posterior = preds*prior

    predictions_preds = np.array([np.argmax(x) for x in preds])
    #predictions_post = np.array([np.argmax(x) for x in posterior])

    #predictions_real = y_test[:,1]
    predictions_real = np.array([np.argmax(x) for x in y_test])


    print(f'accuracy: {accuracy_score(predictions_real, predictions_preds)}')

    cm = confusion_matrix(predictions_real, predictions_preds, labels=[0,1,2,3,4], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['4top', 'ttbar', 'ttbarHiggs', 'ttbarW', 'ttbarZ'])

    disp = disp.plot()
    plt.show()


    #print(f'posterior: {accuracy_score(predictions_real, predictions_post)}')

    # cm = confusion_matrix(predictions_real, predictions_preds, normalize='true')
    # plot_confusion_matrix(cm, classes=['background','4top'])




