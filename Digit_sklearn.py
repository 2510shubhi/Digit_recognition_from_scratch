from sklearn.neural_network import MLPClassifier
import struct
import numpy as np
from sklearn.datasets.mldata import fetch_mldata



if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    # rescale the data, use the traditional train/test split
    X, y = mnist.data / 255., mnist.target
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    clf = MLPClassifier(alpha=0.01,hidden_layer_sizes = (200, 150), random_state = 1, max_iter=10)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print ("Test set score: %f" % clf.score(X_test, y_test))
