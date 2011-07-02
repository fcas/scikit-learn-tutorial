
===============================================================================
Statistical learning: the setting and the estimator object in the scikit-learn
===============================================================================

Datasets
=========

The `scikits.learn` deals with learning information from one or more
datasets that are represented as 2D arrays. They can be understood as a
list of multi-dimensional observations. We say that the first axis of
these arrays is the **samples** axis, while the second is the
**features** axis.

.. topic:: A simple example shipped with the scikit: iris dataset

    ::

        >>> from scikits.learn import datasets
        >>> iris = datasets.load_iris()
        >>> data = iris.data
        >>> data.shape
        (150, 4)

    It is made of 150 observations of irises, each described by 4
    features: their sepal and petal length and width, as detailed in
    `iris.DESCR`.

When the data is not intially in the `(n_samples, n_features)` shape, it
needs to be preprocessed to be used by the scikit.

.. topic:: An example of reshaping data: the digits dataset 

    .. image:: digits_first_image.png
        :align: right
        :scale: 50

    The digits dataset is made of 1797 8x8 images of hand-written
    digits ::

        >>> digits = datasets.load_digits()
        >>> digits.images.shape
        (1797, 8, 8)
        >>> import pylab as pl
        >>> pl.imshow(digits.images[0], cmap=pl.cm.gray_r)

    To use this dataset with the scikit, we transform each 8x8 image in a
    feature vector of length 64 ::

        >>> data = digits.images.reshape((digits.images.shape[0], -1))


Estimators objects
===================

* fitting data

* estimated parameters

* model parameters


