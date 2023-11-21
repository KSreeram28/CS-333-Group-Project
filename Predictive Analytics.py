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

# choose the best classifier model
def choose_best(customScoreLR, modelLR, customScoreSVC, modelSVC, customScoreRF, modelRF, customScoreKNN, modelKNN,
                customScoreXGBoost, modelXGBoost, customScoreNB, modelNB, y_name, X_test, y_test_arr, outcome):
    bestModel, bestModelName, testPerformance, precisionArrTest, recallArrTest, fprArrTest, tprArrTest = "tie/invalid", "tie/invalid", "tie/invalid", "tie/invalid", "tie/invalid", "tie/invalid", "tie/invalid"

    if (
            customScoreLR > customScoreSVC and customScoreLR > customScoreRF and customScoreLR > customScoreKNN and customScoreLR > customScoreXGBoost and customScoreLR > customScoreNB):
        bestModelName = "Logistic Regression"
        testPerformance = performanceMetrics(X_test, y_test, modelLR)
        precisionArrTest, recallArrTest, _ = precision_recall_curve(y_test, modelLR.predict_proba(X_test)[:, 1])
        fprArrTest, tprArrTest, _ = roc_curve(y_test, modelLR.predict_proba(X_test)[:, 1])
        bestModel = modelLR

    elif (
            customScoreSVC > customScoreLR and customScoreSVC > customScoreRF and customScoreSVC > customScoreKNN and customScoreSVC > customScoreXGBoost and customScoreSVC > customScoreNB):
        bestModelName = "Support Vector Classifier"
        testPerformance = performanceMetrics(X_test, y_test, modelSVC)
        precisionArrTest, recallArrTest, _ = precision_recall_curve(y_test, modelSVC.predict_proba(X_test)[:, 1])
        fprArrTest, tprArrTest, _ = roc_curve(y_test, modelSVC.predict_proba(X_test)[:, 1])
        bestModel = modelSVC

    elif (
            customScoreRF > customScoreLR and customScoreRF > customScoreSVC and customScoreRF > customScoreKNN and customScoreRF > customScoreXGBoost and customScoreRF > customScoreNB):
        bestModelName = "Random Forest"
        testPerformance = performanceMetrics(X_test, y_test, modelRF)
        precisionArrTest, recallArrTest, _ = precision_recall_curve(y_test, modelRF.predict_proba(X_test)[:, 1])
        fprArrTest, tprArrTest, _ = roc_curve(y_test, modelRF.predict_proba(X_test)[:, 1])
        bestModel = modelRF

    elif (
            customScoreKNN > customScoreLR and customScoreKNN > customScoreSVC and customScoreKNN > customScoreRF and customScoreKNN > customScoreXGBoost and customScoreKNN > customScoreNB):
        bestModelName = "K-Nearest Neighbors"
        testPerformance = performanceMetrics(X_test, y_test, modelKNN)
        precisionArrTest, recallArrTest, _ = precision_recall_curve(y_test, modelKNN.predict_proba(X_test)[:, 1])
        fprArrTest, tprArrTest, _ = roc_curve(y_test, modelKNN.predict_proba(X_test)[:, 1])
        bestModel = modelKNN

    elif (
            customScoreXGBoost > customScoreLR and customScoreXGBoost > customScoreSVC and customScoreXGBoost > customScoreRF and customScoreXGBoost > customScoreKNN and customScoreXGBoost > customScoreNB):
        bestModelName = "XGBoost"
        testPerformance = performanceMetrics(X_test, y_test, modelXGBoost)
        precisionArrTest, recallArrTest, _ = precision_recall_curve(y_test, modelXGBoost.predict_proba(X_test)[:, 1])
        fprArrTest, tprArrTest, _ = roc_curve(y_test, modelXGBoost.predict_proba(X_test)[:, 1])
        bestModel = modelXGBoost

    elif (
            customScoreNB > customScoreLR and customScoreNB > customScoreSVC and customScoreNB > customScoreRF and customScoreNB > customScoreKNN and customScoreNB > customScoreXGBoost):
        bestModelName = "Naive Bayes"
        testPerformance = performanceMetrics(X_test, y_test, modelNB)
        precisionArrTest, recallArrTest, _ = precision_recall_curve(y_test, modelNB.predict_proba(X_test)[:, 1])
        fprArrTest, tprArrTest, _ = roc_curve(y_test, modelNB.predict_proba(X_test)[:, 1])
        bestModel = modelNB

    else:
        print("Tie/Invalid")
        print(customScoreLR, customScoreSVC, customScoreRF, customScoreKNN, customScoreXGBoost, customScoreNB)
        print("Tie/Invalid")
        print("Tie/Invalid")
        print("Tie/Invalid")

    return bestModel, bestModelName, testPerformance, precisionArrTest, recallArrTest, fprArrTest, tprArrTest

def calc_youden_j(tpr, fpr):
    youdens = tpr + (-1 * fpr)
    index = np.argmax(youdens)
    return index

def shap_values(model, X_train, featureNames):
    explainer = shap.TreeExplainer(model.base_estimator)
    # shap_values = explainer.shap_values(X_train)
    shap_obj = explainer(X_train)
    plt.figure(figsize=(25, 13))
    shap.summary_plot(shap_values=np.take(shap_obj.values, 0, axis=-1),
                      features=X_train,
                      feature_names=featureNames,
                      sort=False, plot_size=None,
                      show=False)
    plt.savefig("features_dotplot.png")

    plt.figure(figsize=(25, 13))
    shap.summary_plot(shap_values=np.take(shap_obj.values, 0, axis=-1),
                      features=X_train,
                      feature_names=featureNames,
                      sort=False, plot_size=None,
                      show=False, plot_type="bar")
    plt.savefig("features_barplot.png")
    return