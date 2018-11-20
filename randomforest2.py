# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge , ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR, LinearSVR
from sklearn.cross_validation import KFold
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
from lightgbm import LGBMRegressor


train = pd.read_csv(r"C:\Users\ling\Desktop\msbd5001\train.csv")
#train.drop(38, inplace=True)
train.drop(['id', 'random_state', ], axis=1, inplace=True)
train = pd.concat([pd.get_dummies(train.iloc[:,0]), train.iloc[:, 1:]], axis=1)
y = train.iloc[:,-1]
train = train.iloc[:, :-1]

features = ['samplePerJobXPenalty', 'n_jobs', 'l1', 'elasticnet']

train['n_jobs'] = train['n_jobs'].apply(lambda x: 8 if x==-1 else x)
train['sampleXClass'] = train['n_samples']*train['n_classes']
#train['max_iterXFlip'] = train['max_iter']*train['flip_y']
train['samplePerJob'] = train['sampleXClass'] * train['n_features'] * train['max_iter'] / train['n_jobs'] 
#train['information_ratio'] = train['n_informative'] / train['n_features']
train['samplePerJobXPenalty'] = train['samplePerJob']
train = train[features]

y = np.log(y)
print(np.corrcoef(train['samplePerJobXPenalty'],y))

#train_test = train.iloc[30:]
#train = train.iloc[:30]
#
#y_test = y[30:]
#y = y[:30]

test = pd.read_csv(r"C:\Users\ling\Desktop\msbd5001\test.csv")
test.drop(['random_state'], axis=1, inplace=True)
test = pd.concat([pd.get_dummies(test.iloc[:,1]), test.iloc[:, 2:]], axis=1)
test['n_jobs'] = test['n_jobs'].apply(lambda x: 8 if x==-1 else x)
test['sampleXClass'] = test['n_samples']*test['n_classes']
#test['max_iterXFlip'] = test['max_iter']*test['flip_y']
test['samplePerJob'] = test['sampleXClass'] * test['n_features'] * test['max_iter'] / test['n_jobs']
#test['information_ratio'] = test['n_informative'] / test['n_features']
test['samplePerJobXPenalty'] = test['samplePerJob']
test = test[features]


rf_clf = RandomForestRegressor(n_estimators=10, max_depth=12, min_samples_split=3, random_state=0, bootstrap=True)
et_clf = ExtraTreesRegressor(n_estimators=100, max_depth=10,min_samples_leaf=1,min_samples_split=3, random_state=0)
gb_clf = GradientBoostingRegressor(learning_rate=0.1, n_estimators=350, max_depth=2, random_state=0)
ada_clf = AdaBoostRegressor(learning_rate=1.0, n_estimators=90, loss='square', base_estimator=DecisionTreeRegressor(max_depth=12, random_state=0), random_state=0)
dt_clf = DecisionTreeRegressor(max_depth=12, random_state=0)
xg_clf = XGBRegressor(learning_rate=0.1, n_estimators=800, max_depth=2, colsample_bytree=0.9, reg_lambda=0.016, random_state=0)
xg_dart_clf = XGBRegressor(booster='dart', n_estimators=400, max_depth=2,reg_alpha=1e-5, reg_lambda=0.019, random_state=0, n_jobs=-1)
lgb_clf = LGBMRegressor(learning_rate=0.01, n_estimators=900, max_depth=3, min_child_samples=1, num_leaves=5,random_state=0)
linear_clf = LinearRegression()
lasso_clf = Lasso()

clf_list = [rf_clf, et_clf, gb_clf, ada_clf, dt_clf,  xg_clf, xg_dart_clf, lgb_clf, linear_clf]
name_list = ['Random Forest', 'Extra Trees', 'Gradient Boosted', 'AdaBoost', 'Decision Tree', 'xgboost', 'xgboost_dart', 'lgb', 'linear']
#clf_list = [linear_clf]
#name_list = ['linear']

def my_loss_func(y_true, y_pred):
    mse = np.sum((np.exp(y_true) - np.exp(y_pred))**2) / len(y_true)
    return mse

my_mse = make_scorer(my_loss_func, greater_is_better=False)
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
for clf, name in zip(clf_list,name_list): 
	scores = cross_val_score(clf, train, y, cv=cv, scoring=my_mse)
	print("Accuracy: %0.2f +/- %0.2f (%s 95%% CI)" % (scores.mean(), scores.std()*2, name))
	clf.fit(train,y)

df_result = pd.DataFrame(np.exp(xg_dart_clf.predict(test)), columns=['Time'])
df_result.index.names = ['Id']
df_result.to_csv(r"C:\Users\ling\Desktop\msbd5001\result2.csv")


## Test Linear
#features = ['samplePerJobXPenalty', 'penalty_transformed']
#train = MinMaxScaler().fit_transform(train[features])
#test = MinMaxScaler().fit_transform(test[features])
#linear_clf = LinearRegression()
#cv = ShuffleSplit(n_splits=10, test_size=0.33, random_state=0)
#scores = cross_val_score(linear_clf, train, np.exp(y), cv=cv)
#linear_clf.fit(train, np.exp(y))
#print("Accuracy: %0.2f +/- %0.2f (%s 95%% CI)" % (scores.mean(), scores.std()*2, name))
#df_result = pd.DataFrame(np.abs(linear_clf.predict(test)), columns=['Time'])
#df_result.index.names = ['Id']
#df_result.to_csv(r"C:\Users\ling\Desktop\msbd5001\result2.csv")



#
#grid_clf = GridSearchCV(XGBRegressor(booster='dart', n_estimators=400, max_depth=2,reg_alpha=1e-5, reg_lambda=0.019, random_state=0, n_jobs=-1), {
#            'reg_lambda': np.arange(0.018,0.02,0.0005)
#        }, scoring=my_mse, verbose=1, cv=cv)
#grid_clf.fit(train,y)
#print(grid_clf.best_params_)
#grid_clf.grid_scores_
#
#
#ntrain=400
#ntest=100
#NFOLDS=5
#kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=0)
#
#def get_oof(clf, x_train, y_train, x_test):
#    oof_train = np.zeros((ntrain,))
#    oof_test = np.zeros((ntest,))
#    oof_test_skf = np.empty((NFOLDS, ntest))
#
#    for i, (train_index, test_index) in enumerate(kf):
#        x_tr = x_train[train_index]
#        y_tr = y_train[train_index]
#        x_te = x_train[test_index]
#
#        clf.fit(x_tr, y_tr)
#
#        oof_train[test_index] = np.exp(clf.predict(x_te))
#        oof_test_skf[i, :] = np.exp(clf.predict(x_test))
#
#    oof_test[:] = oof_test_skf.mean(axis=0)
#    oof_test_full = np.exp(clf.fit(x_train, y_train).predict(x_test))
#    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1), oof_test_full.reshape(-1,1)
#
#rf_oof_train, rf_oof_test, rf_test                = get_oof(rf_clf, train.values, y.values, test.values)
#xg_oof_train, xg_oof_test, xg_test                = get_oof(xg_clf, train.values, y.values, test.values)
#gb_oof_train, gb_oof_test, gb_test                = get_oof(gb_clf, train.values, y.values, test.values)
#xg_dart_oof_train, xg_dart_oof_test, xg_dart_test = get_oof(xg_dart_clf, train.values, y.values, test.values)
#et_oof_train, et_oof_test, et_test                = get_oof(et_clf, train.values, y.values, test.values)
#ada_oof_train, ada_oof_test, ada_test             = get_oof(ada_clf, train.values, y.values, test.values)
#lgb_oof_train, lgb_oof_test, lbg_test             = get_oof(lgb_clf, train.values, y.values, test.values)
#
#
#
#def my_mse_func(y_true, y_pred):
#    mse = np.sum((y_true -y_pred)**2) / len(y_true)
#    return mse
##x_train = np.concatenate((rf_oof_train, xg_oof_train, gb_oof_train, xg_dart_oof_train, et_oof_train, ada_oof_train) , axis=1)
##x_test = np.concatenate((rf_oof_test, xg_oof_test, gb_oof_test, xg_dart_oof_test, et_oof_test, ada_oof_test) , axis=1)
#x_train = np.concatenate((xg_dart_oof_train, xg_oof_train,  ) , axis=1)
#x_test = np.concatenate((xg_dart_oof_test, xg_oof_test,), axis=1)
#x_test_full = np.concatenate((xg_dart_test, xg_test, ), axis=1)
#y_train = np.exp(y)
#
#print("RF-CV: {}".format(my_mse_func(y_train, rf_oof_train.reshape(-1))))
#print("XG-CV: {}".format(my_mse_func(y_train, xg_oof_train.reshape(-1))))
#print("XG-dart-CV: {}".format(my_mse_func(y_train, xg_dart_oof_train.reshape(-1))))
#print("GB-CV: {}".format(my_mse_func(y_train, gb_oof_train.reshape(-1))))
#print("ET-CV: {}".format(my_mse_func(y_train, et_oof_train.reshape(-1))))
#print("ADA-CV: {}".format(my_mse_func(y_train, ada_oof_train.reshape(-1))))
#print("LGB-CV: {}".format(my_mse_func(y_train, lgb_oof_train.reshape(-1))))
#
#import seaborn as sns
#
#
#xg_clf_2 = XGBRegressor(n_estimators=50, max_depth=7, random_state=0)
#dt_clf_2 = DecisionTreeRegressor(random_state=0)
#ridge_clf_2 = Ridge() 
#linear_clf_2 = LinearRegression()
#
#x_train = MinMaxScaler().fit_transform(x_train)
#x_test = MinMaxScaler().fit_transform(x_test)
#x_test_full = MinMaxScaler().fit_transform(x_test_full)
#
#clf_list = [xg_clf_2, dt_clf_2, ridge_clf_2, linear_clf_2]
#name_list = ['xgboost', 'dt', 'ridge', 'linear']
#
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#for clf, name in zip(clf_list,name_list) :
#	scores = cross_val_score(clf, x_train, y_train, cv=cv)
#	print("Accuracy: %0.2f +/- %0.2f (%s 95%% CI)" % (scores.mean(), scores.std()*2, name))
#	clf.fit(x_train,y_train)
#
#
#df_result = pd.DataFrame(ridge_clf_2.predict(x_test), columns=['Time'])
#df_result.index.names = ['Id']
#df_result.to_csv(r"C:\Users\ling\Desktop\msbd5001\result3.csv")
##
##
#grid_clf = GridSearchCV(XGBRegressor(n_estimators=50, max_depth=7, random_state=0), {
#            'n_estimators': np.arange(10,100,10)
#        }, verbose=1, cv=cv)
#grid_clf.fit(x_train,y_train)
#grid_clf.best_params_
#grid_clf.grid_scores_




#
#parameters = {
#        'Random Forest' : {'n_estimators': [5,50]},
#}
#grid_clf = GridSearchCV(ExtraTreesRegressor(n_estimators=45, max_depth=6, random_state=0), {
#            'max_leaf_nodes':np.arange(10,1011,100),
#        }, scoring=my_mse, verbose=2, cv=cv)
#grid_clf.fit(train,y)
#grid_clf.best_params_
#
#grid_clf = GridSearchCV(GradientBoostingRegressor(learning_rate=0.05, n_estimators=700, max_depth=2, min_samples_split=3, random_state=0), {
#            'subsample': [0.95,1]
#        }, scoring=my_mse, verbose=2, cv=cv)
#grid_clf.fit(train,y)
#grid_clf.best_params_
#grid_clf.grid_scores_
#grid_clf = GridSearchCV(XGBRegressor(learning_rate=0.05, n_estimators=840, max_depth=2, reg_lambda=0.015, random_state=0), {
#            'n_estimators':np.arange(800,900,10)
#        }, scoring=my_mse, verbose=1, cv=cv)
#grid_clf.fit(train,y)
#grid_clf.best_params_
#grid_clf.grid_scores_