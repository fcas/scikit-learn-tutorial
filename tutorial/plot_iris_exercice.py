import numpy as np
import pylab as pl
from scikits.learn import datasets, svm

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y!=0]
y = y[y!=0]

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order]

X_train = X[:.9*n_sample]
y_train = y[:.9*n_sample]
X_test = X[:.9*n_sample]
y_test = y[.9*n_sample:]

h = .02 # step size in the mesh

# fit the model
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    pl.figure(fig_num)
    pl.clf()
    pl.set_cmap(pl.cm.Paired)
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_test)

    pl.scatter(X[:,0], X[:,1], c=y, zorder=10)
    pl.scatter(X_test[:,0], X_test[:, 1],
            s=80, facecolors='none', zorder=10)

    pl.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    pl.figure(1, figsize=(4, 3))
    pl.pcolormesh(XX, YY, Z > 0)
    pl.contour(XX, YY, Z, colors=['k', 'k', 'k'], 
              linestyles=['--', '-', '--'], 
              levels=[-.5, 0, .5])

    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)


