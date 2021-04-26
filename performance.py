import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sn
sn.set()




def plot_binary_results(predictions_real, predictions_preds, pos_probs):
    cm = confusion_matrix(predictions_real, predictions_preds, labels=[1, 0], normalize='pred')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['4top', 'background'])

    disp = disp.plot()
    plt.show()

    print(classification_report(predictions_real, predictions_preds))
    fpr, tpr, thresholds = roc_curve(predictions_real, pos_probs)

    plt.plot(fpr, tpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(f'AUC: {np.round(roc_auc_score(predictions_real, pos_probs), 3)}')
    plt.show()



def check_model_performance_as1(X_test, y_test, model):

    output_probs = np.array(model.predict(x = X_test))
    pos_probs = output_probs[:, 1]

    predictions_preds = np.array([np.argmax(x) for x in output_probs])
    predictions_real = y_test[:,1]

    print(f'accuracy: {accuracy_score(predictions_real, predictions_preds)}')

    plot_binary_results(predictions_real, predictions_preds, pos_probs)




def check_model_performance_as3(X_test, y_test, model, threshold):


    output_probs = np.array(model.predict(x = X_test))
    pos_probs = output_probs[:, 0]

    predictions_preds = pos_probs > threshold
    predictions_real = y_test[:, 0]

    print(f'accuracy: {accuracy_score(predictions_real, predictions_preds)}')
    print("Classification Report:\n", classification_report(predictions_real, predictions_preds))

    pos_probs = output_probs
    plot_binary_results(predictions_real, predictions_preds, pos_probs)



def check_model_performance_as2(X_test, y_test, model):
    preds = np.array(model.predict(x = X_test))
    predictions_preds = np.array([np.argmax(x) for x in preds])

    predictions_real = np.array([np.argmax(x) for x in y_test])
    print(f'accuracy: {accuracy_score(predictions_real, predictions_preds)}')

    cm = confusion_matrix(predictions_real, predictions_preds, labels=[0,1,2,3,4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['4top', 'ttbar', 'ttbarHiggs', 'ttbarW', 'ttbarZ'])

    disp = disp.plot()
    plt.show()


    #print(f'posterior: {accuracy_score(predictions_real, predictions_post)}')

    # cm = confusion_matrix(predictions_real, predictions_preds, normalize='true')
    # plot_confusion_matrix(cm, classes=['background','4top'])




