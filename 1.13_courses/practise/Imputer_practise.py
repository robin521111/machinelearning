import numpy as np 
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)

dataset = load_boston()
X_full, Y_full = dataset.data, dataset.target 

n_sample = X_full.shape[0]
n_feature = X_full.shape[1]

estimator = RandomForestRegressor(random_state=0,n_estimators=100)
score = cross_val_score(estimator,X_full,Y_full).mean()
print('Score with the entire dataset = %.2f'% score)

missing_rate = 0.75

n_missing_sample = int(np.floor(n_sample*missing_rate))
missing_samples = np.hstack((np.zeros(n_sample-n_missing_sample,dtype=np.bool),np.ones(n_missing_sample,dtype=np.bool)))
rng.shuffle(missing_samples)

missing_feature = rng.randint(0, n_feature,n_missing_sample)

x_filtered = X_full[~missing_samples,:]
y_filtered = Y_full[~missing_samples]
estimator= RandomForestRegressor(random_state=0,n_estimators=100)
score = cross_val_score(estimator,x_filtered,y_filtered).mean()
print("score without the samples containing missing values = %.2f" % score)

X_missing =X_full.copy()
X_missing[np.where(missing_samples)[0],missing_feature] = 0
y_missing = Y_full.copy()
estimator = Pipeline([("imputer",Imputer(missing_values=0,strategy="mean",axis=0)),("forest",RandomForestRegressor(random_state=0,n_estimators=100))])
score = cross_val_score(estimator,X_missing,y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)
