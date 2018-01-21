Download XGBoost Windows x64 Binaries and Executables
http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/

## Installing the Python Wrapper

Please follow these instructions to prepare XGBoost for use with Python. I am placing xgboost in a directory called xgboost_install_dir but this can be anything.

git clone https://github.com/dmlc/xgboost.git xgboost_install_dir
copy libxgboost.dll (downloaded from this page) into the xgboost_install_dir\python-package\xgboost\ directory
cd xgboost_install_dir\python-package\
python setup.py install

## Using the Python Library

import xgboost
xr = xgboost.XGBRegressor()
xr.fit(X, y)
xr.predict(X_test)