import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# load prepared data
df = pd.read_pickle('prepared_dataframe.pickle')
# print(df.columns.values)
# print(df['emotion'])
# unique_values = df['emotion'].unique()
# print(unique_values)

# Surprised/Neutral Classification

surprised = df[df['emotion'] == 'surprised']
calm = df[df['emotion'] == 'calm']
# print(len(surprised))
# print(len(calm))
valid_emotions = ['calm', 'surprised']
df = df[df['emotion'].isin(valid_emotions)]
# print(df)
# to do: plot mffcc for calm/surprised in order to show difference


# Isolate columns
DatasetDf = df[['emotion', 'mfcc_profile']]
# print(DatasetDf)

# Make values 0, 1

binary_labels = []

for row in DatasetDf['emotion']:
    label = 0.0
    if row == "calm":
        label = 1.0
    binary_labels.append(label)


# Assign features to x and income values to y

np_dataset = DatasetDf.to_numpy()
# print(np_dataset.shape)

# Get features
X_emotions = np_dataset[:, 1]

# Get output values
y_emotions = binary_labels

# print('X_emotions:',X_emotions)
# print('y_emotions:',y_emotions)

# Split data to train/test

X_train, X_test, y_train, y_test = train_test_split(np.vstack(X_emotions), y_emotions, train_size=0.85, test_size=0.15, shuffle=True, random_state=1997)

# Question 4: For better results we can use the GridSearchCV
# which finds the parameters that provide the best model result.
# In our case we managed to increase the accuracy by 12% using GridSearchCV

from sklearn.model_selection import GridSearchCV, ShuffleSplit

# initialize random forest classifier
RandomForest = RandomForestClassifier()

parameters = {'n_estimators': [10, 2000], 'random_state': [3, 2000]}

split_for_grid_search = ShuffleSplit(n_splits=1, test_size=0.15, random_state=1000)
clf = GridSearchCV(RandomForest, param_grid=parameters, cv=split_for_grid_search)
clf.fit(X_train, y_train)

best_params = clf.best_params_

best_clf = RandomForestClassifier(n_estimators=best_params['n_estimators'], random_state=best_params['random_state'])

# print(clf.get_params())

best_clf.fit(X_train, y_train)

# Calculating Accuracy

y_pred = best_clf.predict(X_test)
print("Accuracy of the testing dataset (in %):", metrics.accuracy_score(y_test, y_pred) * 100)


# Creating Confusion Matrix

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels= ['Calm', 'Surprised'],
            yticklabels= ['Calm', 'Surprised'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# Save model
with open("best_clf.pkl", "wb") as fw:
    pickle.dump(best_clf, fw)
