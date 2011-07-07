import numpy as np
import pylab as pl

from scikits.learn import cross_val, datasets, linear_model

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

lasso = linear_model.Lasso()

alphas = np.logspace(-4, -1, 20)

scores = list()
scores_std = list()

for alpha in alphas:
    lasso.alpha = alpha
    scores.append(np.mean(cross_val.cross_val_score(lasso, X, y, n_jobs=-1)))

pl.figure(1, figsize=(2.5, 2))
pl.clf()
pl.axes([.1, .25, .8, .7])
pl.semilogx(alphas, scores)
pl.yticks(())
pl.ylabel('CV score')
pl.xlabel('alpha')
pl.savefig('cv_diabetes.png')

