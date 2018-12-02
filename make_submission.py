# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBRegressor


train = pd.read_csv(r"train.csv")
train.drop(['id', 'random_state', ], axis=1, inplace=True)
train = pd.concat([pd.get_dummies(train.iloc[:,0]), train.iloc[:, 1:]], axis=1)
y = train.iloc[:,-1]
train = train.iloc[:, :-1]

features = ['samplePerJobXPenalty', 'n_jobs', 'l1', 'elasticnet']

train['n_jobs'] = train['n_jobs'].apply(lambda x: 8 if x==-1 else x)
train['sampleXClass'] = train['n_samples']*train['n_classes']
train['samplePerJob'] = train['sampleXClass'] * train['n_features'] * train['max_iter'] / train['n_jobs'] 
train['samplePerJobXPenalty'] = np.log(train['samplePerJob'])
train = train[features]

y = np.log(y)

test = pd.read_csv(r"test.csv")
test.drop(['random_state'], axis=1, inplace=True)
test = pd.concat([pd.get_dummies(test.iloc[:,1]), test.iloc[:, 2:]], axis=1)
test['n_jobs'] = test['n_jobs'].apply(lambda x: 8 if x==-1 else x)
test['sampleXClass'] = test['n_samples']*test['n_classes']
test['samplePerJob'] = test['sampleXClass'] * test['n_features'] * test['max_iter'] / test['n_jobs']
test['samplePerJobXPenalty'] = np.log(test['samplePerJob'])
test = test[features]

xg_clf = XGBRegressor(learning_rate=0.1, n_estimators=800, max_depth=2, colsample_bytree=0.9, reg_lambda=0.016, random_state=0)

clf_list = [xg_clf]
name_list = ['xg_clf']

def my_loss_func(y_true, y_pred):
    mse = np.sum((np.exp(y_true) - np.exp(y_pred))**2) / len(y_true)
    return mse

my_mse = make_scorer(my_loss_func, greater_is_better=False)
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
for clf, name in zip(clf_list,name_list): 
	scores = cross_val_score(clf, train, y, cv=cv, scoring=my_mse)
	print("Accuracy: %0.2f +/- %0.2f (%s 95%% CI)" % (scores.mean(), scores.std()*2, name))
	clf.fit(train,y)

df_result = pd.DataFrame(np.exp(xg_clf.predict(test)), columns=['Time'])
df_result.index.names = ['Id']
df_result.to_csv(r"submission.csv")