=======================================================================================
Supervised learning: predicting an output variable from high-dimensional observations
=======================================================================================

.. topic:: The problem solved in supervised learning

   Supervised learning consists in learning the link between two
   datasets: the observed data `X`, and an external variable `y` that we
   are trying to predict, usually called `target` or `labels`. Most often, 
   `y` is a 1D array of length `n_samples`. 
   
   All supervised estimators in the scikits.learn implement a `fit(X, y)`
   method to fit the model, and a `predict(X)` method that, given
   unlabeled observations `X`, returns predicts the corresponding labels
   `y`.

.. topic:: Vocabulary: classification and regression

   If the prediction task is to classify the observations in a set of
   finite labels, in other words to "name" the objects observed, the task
   is said to be a **classification** task. On the opposite, if the goal
   is to predict a continous target variable, it is said to be a
   **regression** task.

   In the scikits.learn, for classification tasks, `y` is a vector of
   integers.

Nearest neighbor and the curse of dimensionality
=================================================

.. topic:: Classifying irises:

    The iris dataset is a classification task consisting in identifying 3
    different types of irises (Setosa, Versicolour, and Virginica) from
    their petal and sepal length and width::

        >>> import numpy as np
        >>> from scikits.learn import datasets
        >>> iris = datasets.load_iris()
        >>> X = iris.data
        >>> y = iris.target
        >>> np.unique(y)
        array([0, 1, 2])

The simplest possible classifier is the nearesr neighbor: given a new
observation `x_test`, find in the training set -the data used to train
the estimator- the observation with the closest feature vector.

.. topic:: Training set and testing set

   When experimenting with learning algorithm, it is important not to
   test the prediction of an estimator on the data used to fit the
   estimator, as this would not be evaluating the performance of the
   estimator on **new data**. This is why datasets are often split into
   *train* and *test* data.


**KNN (k nearest neighbors) classification example**::

    >>> # Split iris data in train and test data
    >>> # A random permutation, to split the data randomly
    >>> np.random.seed(0)
    >>> indices = np.random.permutation(len(X))
    >>> X_train = X[indices[:-10]]
    >>> y_train = y[indices[:-10]]
    >>> X_test  = X[indices[-10:]]
    >>> y_test  = y[indices[-10:]]
    >>> # Create and fit a nearest-neighbor classifier
    >>> from scikits.learn.neighbors import NeighborsClassifier
    >>> knn = NeighborsClassifier()
    >>> knn.fit(X_train, y_train)
    NeighborsClassifier(n_neighbors=5, window_size=1, algorithm='auto')
    >>> knn.predict(X_test)
    array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])
    >>> y_test
    array([1, 1, 1, 0, 0, 0, 2, 1, 2, 0])


Linear model: from regression to sparsity
==========================================

Support vector machines
========================

Gaussian process: introducing the notion of posterior estimate
===============================================================

