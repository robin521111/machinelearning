import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import pickle
from sklearn.ensemble import RandomForestClassifier as RFC

NUM_DIGITS = 10

# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

# Our goal is to produce fizzbuzz for the numbers 1 to 100. So it would be
# unfair to include these in our training data. Accordingly, the training data
# corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).

trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])

X_train, X_test, y_train, y_test = trX[0:600], trX[600:900], trY[0:600], trY[600:900],

# Learn a simple RF classifier
forest = RFC(n_jobs=2, n_estimators=50, verbose=1, oob_score=True)
forest.fit(X_train, y_train)
# Returns the mean accuracy on the given test data and labels.
print(forest.score(X_test, y_test))
