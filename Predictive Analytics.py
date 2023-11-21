import csv
import pandas as pd

import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import scipy.stats as stat
from sklearn.utils import resample
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import RandomizedSearchCV
import itertools
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import shap
from types import SimpleNamespace
import pickle
from sklearn.preprocessing import label_binarize


def readInData(dataset):
    global df
    global df2
    global colNames
    if (dataset == 1):
        df = pd.read_csv("data_new.csv")
        colNames = df.columns.tolist()
        df = df.dropna()
        print(df)
        colNames = df.columns.tolist()
        print(colNames)
        print(df.info())

def HPOptimize(clf, par, trainX, trainY, valX, valY, name, lossChoice):
    # Remember, this function returns a function -- yay for functional progrmaming, finally get to use the stuff I learned in that class lmao.
    customLoss, proba = choose_custom_loss(lossChoice)

    cv = RandomizedSearchCV(clf, par, cv=9, verbose=4,
                            scoring=make_scorer(customLoss, greater_is_better=True, needs_proba=proba), refit=True,
                            n_jobs=-1, n_iter=1)
    print(cv)
    # cv = GridSearchCV(clf, par, cv=5,  verbose=4, scoring=make_scorer(customLoss, greater_is_better=True, needs_proba=proba), refit=True, n_jobs=-1)
    search = cv.fit(trainX, trainY)
    topModel = search.best_estimator_
    topParams = search.best_params_

    paramDict = search.cv_results_

    calibrated_clf = CalibratedClassifierCV(base_estimator=topModel, cv=9, method='isotonic')
    cal_model = calibrated_clf.fit(trainX, trainY)

    # Test Set Performance: AUROC, precision, recall
    performance = performanceMetrics(valX, valY, cal_model)
    customScore = performance[0]

    y_score = cal_model.predict_proba(valX)
    precisionArr, recallArr, _ = precision_recall_curve(valY, y_score[:, 1])

    fprArr, tprArr, _ = roc_curve(valY, y_score[:, 1])

    return cal_model, topParams, customScore, performance, precisionArr, recallArr, fprArr, tprArr, name, paramDict