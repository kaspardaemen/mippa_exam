import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from network import build_neural_network, build_conv_network
from preprocessing import get_data_a
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# MAIN
# =============================================================================

n_epochs = 100
dropout = 0
file = 'TrainingValidationData_200k_shuffle.csv'
events_only = True

#classes list: [False, True]
data, labels, df = get_data_a(file)

#don't use met data for conv. network
indices = range(data.shape[1])
if(events_only):
    indices = indices[2:]
data = data[:,indices].reshape(-1,19,4)


X_train, X_test, y_train, y_test = train_test_split(data[indices], labels[indices], test_size=0.2, random_state=42)

model = build_conv_network(X_train.shape[1], dropout)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=n_epochs, batch_size=128, validation_data=(X_test, y_test))

# preds = np.array(model.predict(x = X_test))
# prior = np.array([0.8, 0.2])
#
# posterior = preds*prior
#
# predictions_preds = np.array([np.argmax(x) for x in preds])
# predictions_post = np.array([np.argmax(x) for x in posterior])
#
# predictions_real = y_test[:,1]
#
# accuracy_score(predictions_real, predictions_preds)
# accuracy_score(predictions_real, predictions_post)

hist_df = pd.DataFrame(hist.history)
hist_df['epoch'] = hist_df.index + 1
vals = ['accuracy', 'val_accuracy']

sns.lineplot(data=hist_df[['accuracy', 'val_accuracy']])
plt.show()

sns.lineplot(data=hist_df[['loss', 'val_loss']])
plt.show()
