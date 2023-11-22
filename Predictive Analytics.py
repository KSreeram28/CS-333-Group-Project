
import pandas as pd

from datetime import datetime
import csv

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report, accuracy_score, \
    average_precision_score, precision_score, recall_score, make_scorer, roc_curve, precision_recall_curve, \
    average_precision_score, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing, datasets
from sklearn.preprocessing import StandardScaler
import random
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


def readInData(dataset):
    global df
    global colNames
    df = pd.read_csv(dataset)
    colNames = df.columns.tolist()
    df = df.dropna()
    print(df)
    print(colNames)
    print(df.info())
    percentile_55 = df["risk_score_t"].quantile(0.55)
    percentile_97 = df["risk_score_t"].quantile(0.97)
    df["Referral"] = np.select([df["risk_score_t"] > percentile_97,
                                  (df["risk_score_t"] > percentile_55) & (df["risk_score_t"] <= percentile_97),
                                  df["risk_score_t"] <= percentile_55],
                                 [2, 1, 0],
                                 default=0)

    colNames = df.columns.tolist()

    df = df.drop(columns = ["risk_score_t", "Unnamed: 0"])
    print(df)
    print("missing data",np.sum(df.isnull().sum()))



def HPOptimize(clf, par, trainX, trainY, valX, valY, name, lossChoice):
    # Remember, this function returns a function -- yay for functional progrmaming, finally get to use the stuff I learned in that class lmao.
    customLoss, proba = choose_custom_loss(lossChoice)
    cv = RandomizedSearchCV(clf, par, cv=5, verbose=4,
                            scoring=make_scorer(customLoss, greater_is_better=True, needs_proba=proba), refit=True,
                            n_jobs=-1, n_iter=1)
    print(cv)
    # cv = GridSearchCV(clf, par, cv=5,  verbose=4, scoring=make_scorer(customLoss, greater_is_better=True, needs_proba=proba), refit=True, n_jobs=-1)
    search = cv.fit(trainX, trainY)
    topModel = search.best_estimator_
    topParams = search.best_params_

    paramDict = search.cv_results_

    calibrated_clf = CalibratedClassifierCV(base_estimator=topModel, cv=5, method='isotonic')
    cal_model = calibrated_clf.fit(trainX, trainY)

    # Test Set Performance: AUROC, precision, recall
    performance = performanceMetrics(valX, valY, cal_model)
    customScore = performance[0]

    valY = label_binarize(valY, classes=np.arange(3))
    y_score = cal_model.predict_proba(valX)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(valY[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    #micro
    fpr["micro"], tpr["micro"], _ = roc_curve(valY.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))


    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 3

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #fprArr, tprArr, _ = roc_curve(valY, y_score[:, 1])

    return cal_model, topParams, customScore, performance, fpr, tpr, name, paramDict, roc_auc

# choose the best classifier model
def choose_best(customScoreLR, modelLR, customScoreSVC, modelSVC, customScoreRF, modelRF, customScoreKNN, modelKNN,
                customScoreXGBoost, modelXGBoost, customScoreNB, modelNB, y_name, X_test, y_test, outcome):
    bestModel, bestModelName, testPerformance, fprArrTest, tprArrTest, rocaucArrTest = "tie/invalid", "tie/invalid", "tie/invalid", "tie/invalid", "tie/invalid", "tie/invalid"

    if (
            customScoreLR > customScoreSVC and customScoreLR > customScoreRF and customScoreLR > customScoreKNN and customScoreLR > customScoreXGBoost and customScoreLR > customScoreNB):
        bestModelName = "Logistic Regression"
        testPerformance = performanceMetrics(X_test, y_test, modelLR)
        y_test = label_binarize(y_test, classes=np.arange(3))

        y_score = modelLR.predict_proba(X_test)
        fprArrTest = dict()
        tprArrTest = dict()
        rocaucArrTest = dict()
        for i in range(3):
            fprArrTest[i], tprArrTest[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            rocaucArrTest[i] = auc(fprArrTest[i], tprArrTest[i])

        # micro
        fprArrTest["micro"], tprArrTest["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        rocaucArrTest["micro"] = auc(fprArrTest["micro"], tprArrTest["micro"])

        # macro
        all_fpr = np.unique(np.concatenate([fprArrTest[i] for i in range(3)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += np.interp(all_fpr, fprArrTest[i], tprArrTest[i])

        # Finally average it and compute AUC
        mean_tpr /= 3

        fprArrTest["macro"] = all_fpr
        tprArrTest["macro"] = mean_tpr
        rocaucArrTest["macro"] = auc(fprArrTest["macro"], tprArrTest["macro"])
        bestModel = modelLR

    elif (
            customScoreSVC > customScoreLR and customScoreSVC > customScoreRF and customScoreSVC > customScoreKNN and customScoreSVC > customScoreXGBoost and customScoreSVC > customScoreNB):
        bestModelName = "Support Vector Classifier"
        testPerformance = performanceMetrics(X_test, y_test, modelSVC)
        y_test = label_binarize(y_test, classes=np.arange(3))

        y_score = modelSVC.predict_proba(X_test)
        fprArrTest = dict()
        tprArrTest = dict()
        rocaucArrTest = dict()
        for i in range(3):
            fprArrTest[i], tprArrTest[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            rocaucArrTest[i] = auc(fprArrTest[i], tprArrTest[i])

        # micro
        fprArrTest["micro"], tprArrTest["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        rocaucArrTest["micro"] = auc(fprArrTest["micro"], tprArrTest["micro"])

        # macro
        all_fpr = np.unique(np.concatenate([fprArrTest[i] for i in range(3)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += np.interp(all_fpr, fprArrTest[i], tprArrTest[i])

        # Finally average it and compute AUC
        mean_tpr /= 3

        fprArrTest["macro"] = all_fpr
        tprArrTest["macro"] = mean_tpr
        rocaucArrTest["macro"] = auc(fprArrTest["macro"], tprArrTest["macro"])
        bestModel = modelSVC

    elif (
            customScoreRF > customScoreLR and customScoreRF > customScoreSVC and customScoreRF > customScoreKNN and customScoreRF > customScoreXGBoost and customScoreRF > customScoreNB):
        bestModelName = "Random Forest"
        testPerformance = performanceMetrics(X_test, y_test, modelRF)
        y_test = label_binarize(y_test, classes=np.arange(3))

        y_score = modelRF.predict_proba(X_test)
        fprArrTest = dict()
        tprArrTest = dict()
        rocaucArrTest = dict()
        for i in range(3):
            fprArrTest[i], tprArrTest[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            rocaucArrTest[i] = auc(fprArrTest[i], tprArrTest[i])

        # micro
        fprArrTest["micro"], tprArrTest["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        rocaucArrTest["micro"] = auc(fprArrTest["micro"], tprArrTest["micro"])

        # macro
        all_fpr = np.unique(np.concatenate([fprArrTest[i] for i in range(3)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += np.interp(all_fpr, fprArrTest[i], tprArrTest[i])

        # Finally average it and compute AUC
        mean_tpr /= 3

        fprArrTest["macro"] = all_fpr
        tprArrTest["macro"] = mean_tpr
        rocaucArrTest["macro"] = auc(fprArrTest["macro"], tprArrTest["macro"])
        bestModel = modelRF

    elif (
            customScoreKNN > customScoreLR and customScoreKNN > customScoreSVC and customScoreKNN > customScoreRF and customScoreKNN > customScoreXGBoost and customScoreKNN > customScoreNB):
        bestModelName = "K-Nearest Neighbors"
        testPerformance = performanceMetrics(X_test, y_test, modelKNN)
        y_test = label_binarize(y_test, classes=np.arange(3))

        y_score = modelKNN.predict_proba(X_test)
        fprArrTest = dict()
        tprArrTest = dict()
        rocaucArrTest = dict()
        for i in range(3):
            fprArrTest[i], tprArrTest[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            rocaucArrTest[i] = auc(fprArrTest[i], tprArrTest[i])

        # micro
        fprArrTest["micro"], tprArrTest["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        rocaucArrTest["micro"] = auc(fprArrTest["micro"], tprArrTest["micro"])

        # macro
        all_fpr = np.unique(np.concatenate([fprArrTest[i] for i in range(3)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += np.interp(all_fpr, fprArrTest[i], tprArrTest[i])

        # Finally average it and compute AUC
        mean_tpr /= 3

        fprArrTest["macro"] = all_fpr
        tprArrTest["macro"] = mean_tpr
        rocaucArrTest["macro"] = auc(fprArrTest["macro"], tprArrTest["macro"])
        bestModel = modelKNN

    elif (
            customScoreXGBoost > customScoreLR and customScoreXGBoost > customScoreSVC and customScoreXGBoost > customScoreRF and customScoreXGBoost > customScoreKNN and customScoreXGBoost > customScoreNB):
        bestModelName = "XGBoost"
        testPerformance = performanceMetrics(X_test, y_test, modelXGBoost)
        y_test = label_binarize(y_test, classes=np.arange(3))

        y_score = modelXGBoost.predict_proba(X_test)
        fprArrTest = dict()
        tprArrTest = dict()
        rocaucArrTest = dict()
        for i in range(3):
            fprArrTest[i], tprArrTest[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            rocaucArrTest[i] = auc(fprArrTest[i], tprArrTest[i])

        # micro
        fprArrTest["micro"], tprArrTest["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        rocaucArrTest["micro"] = auc(fprArrTest["micro"], tprArrTest["micro"])

        # macro
        all_fpr = np.unique(np.concatenate([fprArrTest[i] for i in range(3)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += np.interp(all_fpr, fprArrTest[i], tprArrTest[i])

        # Finally average it and compute AUC
        mean_tpr /= 3

        fprArrTest["macro"] = all_fpr
        tprArrTest["macro"] = mean_tpr
        rocaucArrTest["macro"] = auc(fprArrTest["macro"], tprArrTest["macro"])
        bestModel = modelXGBoost

    elif (
            customScoreNB > customScoreLR and customScoreNB > customScoreSVC and customScoreNB > customScoreRF and customScoreNB > customScoreKNN and customScoreNB > customScoreXGBoost):
        bestModelName = "Naive Bayes"
        testPerformance = performanceMetrics(X_test, y_test, modelNB)
        y_test = label_binarize(y_test, classes=np.arange(3))

        y_score = modelNB.predict_proba(X_test)
        fprArrTest = dict()
        tprArrTest = dict()
        rocaucArrTest = dict()
        for i in range(3):
            fprArrTest[i], tprArrTest[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            rocaucArrTest[i] = auc(fprArrTest[i], tprArrTest[i])

        # micro
        fprArrTest["micro"], tprArrTest["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        rocaucArrTest["micro"] = auc(fprArrTest["micro"], tprArrTest["micro"])

        # macro
        all_fpr = np.unique(np.concatenate([fprArrTest[i] for i in range(3)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += np.interp(all_fpr, fprArrTest[i], tprArrTest[i])

        # Finally average it and compute AUC
        mean_tpr /= 3

        fprArrTest["macro"] = all_fpr
        tprArrTest["macro"] = mean_tpr
        rocaucArrTest["macro"] = auc(fprArrTest["macro"], tprArrTest["macro"])
        bestModel = modelNB

    else:
        print("Tie/Invalid")
        print(customScoreLR, customScoreSVC, customScoreRF, customScoreKNN, customScoreXGBoost, customScoreNB)
        print("Tie/Invalid")
        print("Tie/Invalid")
        print("Tie/Invalid")
    print(bestModel)
    print(fprArrTest)
    print(tprArrTest)
    print(rocaucArrTest)
    return bestModel, bestModelName, testPerformance, fprArrTest, tprArrTest, rocaucArrTest



# def shap_values(model, X_train, featureNames):
#     explainer = shap.TreeExplainer(model.base_estimator)
#     # shap_values = explainer.shap_values(X_train)
#     shap_obj = explainer(X_train)
#     plt.figure(figsize=(25, 13))
#     shap.summary_plot(shap_values=np.take(shap_obj.values, 0, axis=-1),
#                       features=X_train,
#                       feature_names=featureNames,
#                       sort=False, plot_size=None,
#                       show=False)
#     plt.savefig("features_dotplot.png")
#
#     plt.figure(figsize=(25, 13))
#     shap.summary_plot(shap_values=np.take(shap_obj.values, 0, axis=-1),
#                       features=X_train,
#                       feature_names=featureNames,
#                       sort=False, plot_size=None,
#                       show=False, plot_type="bar")
#     plt.savefig("features_barplot.png")
#     return

def nameLoss(lossChoice):  # Eventually pass in case, that is associated with which loss function
    if (lossChoice == 1):
        return "AUCROC AUPRC Geometric Mean"
    elif (lossChoice == 2):
        return "AUCROC"
    elif (lossChoice == 3):
        return "AUPRC"
    elif (lossChoice == 4):  # precision
        return "Precision"
    elif (lossChoice == 5):  # sensitivity, aka recall, aka TPR
        return "Recall"
    elif (lossChoice == 6):  # specificity
        return "Specificity"
    elif (lossChoice == 7):  # ppv
        return "PPV"
    elif (lossChoice == 8):  # npv
        return "NPV"
    return


def performanceMetrics(X, y, model):
    y_proba = model.predict_proba(X)
    y_pred = model.predict(X)

    auroc = roc_auc_score(y, y_proba, multi_class="ovr")

    return [auroc]


def choose_custom_loss(lossChoice):
    if (lossChoice == 1):
        def customLoss(true, pred):
            return np.sqrt(roc_auc_score(true, pred) * average_precision_score(true, pred))

        proba = True

    elif (lossChoice == 2):
        def customLoss(true, pred):
            return roc_auc_score(true, pred, multi_class="ovr")

        proba = True

    elif (lossChoice == 3):
        def customLoss(true, pred):
            return average_precision_score(true, pred)

        proba = False

    elif (lossChoice == 4):  # precision
        def customLoss(true, pred):
            return precision_score(true, pred)

        proba = False

    elif (lossChoice == 5):  # sensitivity, aka recall, aka TPR
        def customLoss(true, pred):
            return recall_score(true, pred)

        proba = False

    elif (lossChoice == 6):  # specificity
        def customLoss(true, pred):
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
            specificity = tn / (tn + fp)
            return specificity

        proba = False

    elif (lossChoice == 7):  # ppv
        def customLoss(true, pred):
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
            ppv = tp / (tp + fp)
            return ppv

        proba = False

    elif (lossChoice == 8):  # npv
        def customLoss(true, pred):
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
            npv = tn / (tn + fn)
            return npv

        proba = False

    return customLoss, proba

def customLossNB(true, X_val, modelNB, lossChoice):
    if (lossChoice == 1):
        pred = modelNB.predict_proba(X_val)[:, 1]
        return np.sqrt(roc_auc_score(true, pred) * average_precision_score(true, pred))

    elif (lossChoice == 2):
        pred = modelNB.predict_proba(X_val)
        return roc_auc_score(true, pred, multi_class = "ovr")

    elif (lossChoice == 3):
        pred = modelNB.predict(X_val)
        return average_precision_score(true, pred)

    elif (lossChoice == 4):  # precision
        pred = modelNB.predict(X_val)
        return precision_score(true, pred)

    elif (lossChoice == 5):  # sensitivity, aka recall, aka TPR
        pred = modelNB.predict(X_val)
        return recall_score(true, pred)

    elif (lossChoice == 6):  # specificity
        pred = modelNB.predict(X_val)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        specificity = tn / (tn + fp)
        return specificity

    elif (lossChoice == 7):  # ppv
        pred = modelNB.predict(X_val)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        ppv = tp / (tp + fp)
        return ppv

    elif (lossChoice == 8):  # npv
        pred = modelNB.predict(X_val)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        npv = tn / (tn + fn)
        return npv

    return

def plot_master(fprArrList, tprArrList, rocaucArrList, nameList, lossFunction, outcome,
                performanceList):
    if (len(fprArrList) != len(tprArrList)):
        return "Lists passed are of unequal length"

    if (len(nameList) < 2):
        figureType = "Test"
    else:
        figureType = "Validation"

    color = ['#e58e23', 'b', 'g', 'r', 'c', 'm', 'y']

    plt.figure(figsize=(14, 7))
    for i in range(len(fprArrList)):
        fpr = fprArrList[i]
        tpr = tprArrList[i]
        roc=rocaucArrList[i]
        for j in range(3):
            if j == 0:
                plt.plot(fpr[j], tpr[j], lw=2, label = "ROC curve of class Not at Risk (area = {0:0.2f})".format(roc[j]))
            elif j == 1:
                plt.plot(fpr[j], tpr[j], lw=2, label = "ROC curve of class Screened Risk (area = {0:0.2f})".format(roc[j]))
            elif j == 2:
                plt.plot(fpr[j], tpr[j], lw=2, label = "ROC curve of class High Risk (area = {0:0.2f})".format(roc[j]))



        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )
        plt.suptitle("Loss Function: " + lossFunction + "           Outcome: " + outcome)


        #if (figureType == "Test"):
            #ix = calc_youden_j(tprArrList[i], fprArrList[i])
            #axes[0].plot([fprArrList[i][ix], fprArrList[i][ix]], [fprArrList[i][ix], tprArrList[i][ix]], color='black',
                        # label="Youden's J statistic = %.4f" % (tprArrList[i][ix] - fprArrList[i][ix]), zorder=3)
            #axes[0].scatter(fprArrList[i][ix], tprArrList[i][ix], color='black', marker='+', s=300, zorder=10)

        plt.title("AUROC for " + nameList[0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim([-0.01, 1.01])
        plt.xlim([-0.01, 1.01])
        plt.legend(loc='lower right', fontsize='small')
    plt.savefig("PW_ROC_figure_{}_{}_{}_train.png".format(figureType, lossFunction, outcome))
    return 0

def plot_all(fprArrList, tprArrList, rocaucArrList, nameList, lossFunction, outcome,
                performanceList):
    if (len(fprArrList) != len(tprArrList)):
        return "Lists passed are of unequal length"

    if (len(nameList) < 2):
        figureType = "Test"
    else:
        figureType = "Validation"

    color = ['#e58e23', 'b', 'g', 'r', 'c', 'm', 'y']

    plt.figure(figsize=(14, 7))
    for i in range(len(fprArrList)):

        fpr = fprArrList[i]
        tpr = tprArrList[i]
        roc=rocaucArrList[i]
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label= nameList[i] + " AUC = {0:0.2f}".format(roc["macro"]),
            color=color[i],
            linestyle=":",
            linewidth=4,
        )

        plt.suptitle("Loss Function: " + lossFunction + "           Outcome: " + outcome)


        plt.title("Macro Average ROCs ")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim([-0.01, 1.01])
        plt.xlim([-0.01, 1.01])
        plt.legend(loc='lower right', fontsize='small')
    plt.savefig("PW_ROC_figure_{}_{}_{}_comparison.png".format(figureType, lossFunction, outcome))
    return 0



lossChoice = 2
readInData("balanced_df.csv")
y = df["Referral"]
X = df.drop(columns = ["Referral"])
myName = colNames

#sparseColumnRemoval(X, y, nameLoss(lossChoice))
featureNames = X.columns.tolist()
print(featureNames)

# 0.15 test, 0.15 val, 0.7 train

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.11111, random_state=42, stratify=y_train_val)



y_test_arr = y_test.to_numpy()

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
myOutcome = "Risk Assessment"

#SAGA Penalty
modelLR, paramsLR, customScoreLR, performanceLR, fprArrLR, tprArrLR, nameLR, paramDictLR, rocaucArrLR = HPOptimize(
    LogisticRegression(solver='saga', max_iter=500, l1_ratio=0.5),
    {'penalty': ('l1', 'l2', 'elasticnet', 'none'), 'C': np.linspace(0.1, 10, 50)}, X_train, y_train, X_val, y_val,
    "Logistic Regression", lossChoice)

# SVC
modelSVC, paramsSVC, customScoreSVC, performanceSVC, fprArrSVC, tprArrSVC, nameSVC, paramDictSVC, rocaucArrSVC = HPOptimize(
    SVC(probability=True),
    {'class_weight': [{1: w} for w in np.linspace(5, 50, 30)], 'kernel': ('poly', 'rbf', 'sigmoid'),
     'gamma': np.logspace(-5, 2, 9)}, X_train, y_train, X_val, y_val, "Support Vector Classifier", lossChoice)

# #RF
modelRF, paramsRF, customScoreRF, performanceRF, fprArrRF, tprArrRF, nameRF, paramDictRF, rocaucArrRF = HPOptimize(
    RandomForestClassifier(), {'n_estimators': np.linspace(100, 1000, 100, dtype=int, endpoint=False)}, X_train,
    y_train, X_val, y_val, "Random Forest", lossChoice)

# #KNN
modelKNN, paramsKNN, customScoreKNN, performanceKNN, fprArrKNN, tprArrKNN, nameKNN, paramDictKNN, rocaucArrKNN = HPOptimize(
    KNeighborsClassifier(), {'n_neighbors': np.linspace(1, 20, 10, dtype=int)}, X_train, y_train, X_val, y_val,
    "K-Nearest Neighbors", lossChoice)

# XGBoost
modelXGBoost, paramsXGBoost, customScoreXGBoost, performanceXGBoost, fprArrXGBoost, tprArrXGBoost, nameXGBoost, paramDictXGBoost, rocaucArrXGBoost = HPOptimize(
    XGBClassifier(),
    {'max_depth': range(2, 10, 10), 'n_estimators': range(60, 220, 40), 'learning_rate': [0.1, 0.01, 0.05]}, X_train,
    y_train, X_val, y_val, "XGBoost", lossChoice)

# Naive Bayes
gnb = OneVsRestClassifier(GaussianNB())
pre_cal_NB = gnb.fit(X_train, y_train)

calibrated_clf = CalibratedClassifierCV(base_estimator=pre_cal_NB, cv=9, method='isotonic')
modelNB = calibrated_clf.fit(X_train, y_train)
customScoreNB, performanceNB, = customLossNB(y_val, X_val, modelNB, lossChoice), performanceMetrics(X_val, y_val,
                                                                                                    modelNB)
y_val = label_binarize(y_test, classes=np.arange(3))
y_score = modelNB.predict_proba(X_val)

fprArrNB = dict()
tprArrNB = dict()
rocaucArrNB = dict()


for i in range(3):
    print(y_val[:, i], y_score[:, i])
    fprArrNB[i], tprArrNB[i], _ = roc_curve(y_val[:, i], y_score[:, i])
    rocaucArrNB[i] = auc(fprArrNB[i], tprArrNB[i])

#micro
fprArrNB["micro"], tprArrNB["micro"], _ = roc_curve(y_val.ravel(), y_score.ravel())
rocaucArrNB["micro"] = auc(fprArrNB["micro"], tprArrNB["micro"])

#macro
all_fprNB = np.unique(np.concatenate([fprArrNB[i] for i in range(3)]))


mean_tprNB = np.zeros_like(all_fprNB)
for i in range(3):
    mean_tprNB += np.interp(all_fprNB, fprArrNB[i], tprArrNB[i])
# Finally average it and compute AUC
mean_tprNB /= 3

fprArrNB["macro"] = all_fprNB
tprArrNB["macro"] = mean_tprNB
rocaucArrNB["macro"] = auc(fprArrNB["macro"], tprArrNB["macro"])
nameNB = "Naive Bayes"

modelList = [modelLR, modelSVC, modelRF, modelKNN, modelXGBoost, modelNB]
paramsList = [paramsLR, paramsSVC, paramsRF, paramsKNN, paramsXGBoost]
customScoreList = [customScoreLR, customScoreSVC, customScoreRF, customScoreKNN, customScoreXGBoost, customScoreNB]
print(customScoreList)
performanceList = [performanceLR, performanceSVC, performanceRF, performanceKNN, performanceXGBoost, performanceNB]
paramDictList = [paramDictLR, paramDictSVC, paramDictRF, paramDictKNN, paramDictXGBoost]
fprArrList = [fprArrLR, fprArrSVC, fprArrRF, fprArrKNN, fprArrXGBoost, fprArrNB]
tprArrList = [tprArrLR, tprArrSVC, tprArrRF, tprArrKNN, tprArrXGBoost, tprArrNB]
rocaucArrList = [rocaucArrLR, rocaucArrSVC, rocaucArrRF, rocaucArrKNN, rocaucArrXGBoost, rocaucArrNB]
nameList = [nameLR, nameSVC, nameRF, nameKNN, nameXGBoost, nameNB]

# Logic to choose which model to use on test set
bestModel, bestModelName, testPerformance, fprArrTest, tprArrTest, rocaucArrTest = choose_best(
    customScoreLR, modelLR, customScoreSVC, modelSVC,
    customScoreRF, modelRF, customScoreKNN, modelKNN, customScoreXGBoost,
    modelXGBoost, customScoreNB, modelNB, myName, X_test, y_test, nameLoss(lossChoice))
filename = bestModelName + "_model.sav"
pickle.dump(bestModel.base_estimator, open(filename, "wb"))

#shap_values(bestModel, X_train, featureNames)
print(bestModel.base_estimator.feature_importances_)
print("Saving Test Predicitions")
gt_preds = np.stack((y_test, bestModel.predict_proba(X_test)[:, 1]), axis=1)
np.savetxt("gt_preds_test_{}_{}.csv".format(myOutcome, nameLoss(lossChoice)), gt_preds, fmt='%s', delimiter=',')

# output_file("allresults.txt", "a", nameLoss(lossChoice), myOutcome, paramsList, performanceList, paramDictList)

# output_file("lastRun.txt", "w", nameLoss(lossChoice), myOutcome, paramsList, performanceList, paramDictList)

# Validation Set Plots
plot_all(fprArrList, tprArrList, rocaucArrList, nameList, nameLoss(lossChoice), myOutcome,
            performanceList)

# Test Set Plots
plot_master([fprArrTest], [tprArrTest], [rocaucArrTest], [bestModelName], nameLoss(lossChoice),
           myOutcome, [testPerformance])

model_dict = {}
for i in range(len(modelList)):
    model_dict[nameList[i]] = modelList[i]