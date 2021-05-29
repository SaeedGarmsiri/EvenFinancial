import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.metrics as metrics
import warnings
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


class LogisticModel:

    def data_splitting(self, df):
        df = df.drop(['lead_uuid', 'offer_id'], axis=1)
        x = df.drop('clicked', axis=1)
        y = df['clicked']
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13, shuffle=True)
        return X_train, X_test, y_train, y_test

    def feature_standardization(self, X_train, X_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        return X_train, X_test

    def pca(self, X_train, X_test):
        pca = PCA()
        X_train = pd.DataFrame(pca.fit_transform(X_train))
        X_test = pd.DataFrame(pca.transform(X_test))

        return X_train, X_test

    def feature_selection(self, X_train, X_test):
        sel = SelectFromModel(xgb.XGBClassifier(n_estimators=100))
        sel.fit(X_train, y_train)
        selected_feat = X_train.columns[(sel.get_support())]
        return X_train[selected_feat], X_test[selected_feat]

    def under_smapling(self, X, y, observations=range(1, 65)):
        scores = []
        for n in observations:
            rus = RandomUnderSampler(random_state=0)
            rus.fit(X, y)
            X_resampled, y_resampled = rus.fit_resample(X, y)
        return X_resampled, y_resampled

    def over_smapling(self, X_train, y_train):
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
        return X_train_res, y_train_res

    def simple_logistic_model(self, X_train, X_test, y_train, y_test ):
        lg1 = LogisticRegression(random_state=13, class_weight=None)
        lg1.fit(X_train, y_train)
        y_pred = lg1.predict(X_test)
        print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
        print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
        print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
        print(f'Recall score: {recall_score(y_test, y_pred)}')

    def weighted_logistic_model(self, X_train, X_test, y_train, y_test):
        class_zero_weight = round((y_train.value_counts()[1] / y_train.value_counts()[0]) * 100, 0).astype(int)
        class_one_weight = 100 - class_zero_weight

        w = {0: class_zero_weight, 1: class_one_weight}
        model = LogisticRegression(random_state=13, class_weight=w)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
        print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
        print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
        print(f'Recall score: {recall_score(y_test, y_pred)}')

    def weight_grid_search(self, X_train, y_train):
        w = [{0: 1000, 1: 100}, {0: 1000, 1: 10}, {0: 1000, 1: 1.0},
             {0: 500, 1: 1.0}, {0: 400, 1: 1.0}, {0: 300, 1: 1.0}, {0: 200, 1: 1.0},
             {0: 150, 1: 1.0}, {0: 100, 1: 1.0}, {0: 99, 1: 1.0}, {0: 10, 1: 1.0},
             {0: 0.01, 1: 1.0}, {0: 0.01, 1: 10}, {0: 0.01, 1: 100},
             {0: 0.001, 1: 1.0}, {0: 0.005, 1: 1.0}, {0: 1.0, 1: 1.0},
             {0: 1.0, 1: 0.1}, {0: 10, 1: 0.1}, {0: 100, 1: 0.1},
             {0: 10, 1: 0.01}, {0: 1.0, 1: 0.01}, {0: 1.0, 1: 0.001}, {0: 1.0, 1: 0.005},
             {0: 1.0, 1: 10}, {0: 1.0, 1: 99}, {0: 1.0, 1: 100}, {0: 1.0, 1: 150},
             {0: 1.0, 1: 200}, {0: 1.0, 1: 300}, {0: 1.0, 1: 400}, {0: 1.0, 1: 500},
             {0: 1.0, 1: 1000}, {0: 10, 1: 1000}, {0: 100, 1: 1000}, 'balanced']
        crange = np.arange(0.5, 20.0, 0.5)
        hyperparam_grid = {"class_weight": w
            , "penalty": ["l1", "l2", "elasticnet"]
            , "C": crange
            , "fit_intercept": [True, False], "l1_ratio": [0.5, 0.8, 1]}

        model = LogisticRegression(random_state=13, solver='saga')
        k_fold = KFold(n_splits=5, shuffle=True)
        grid = RandomizedSearchCV(model, hyperparam_grid, scoring="roc_auc", cv=k_fold, n_jobs=-1, refit=True)
        random_under_sampler = RandomUnderSampler(sampling_strategy='not majority')
        X_train, y_train = random_under_sampler.fit_resample(X_train, y_train)
        grid.fit(X_train, y_train)
        print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

    def to_labels(self, pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')

    def model_weighted_grid(self, X_train, X_test, y_train, y_test):
        model = LogisticRegression(class_weight={0: 1.0, 1: 100},
                                   penalty='l2',
                                   fit_intercept=True,
                                   C=4.0,
                                   solver='liblinear')
        k_fold = KFold(n_splits=5, shuffle=True)
        random_under_sampler = RandomUnderSampler(sampling_strategy='not majority')
        X_train, y_train = random_under_sampler.fit_resample(X_train, y_train)
        weights = [{0: 1.0, 1: 100},
                   {0: 1, 1: 50},
                   {0: 2.0, 1: 100},
                   {0: 5.0, 1: 100},
                   {0: 10, 1: 100},
                   {0: 30, 1: 100}]

        params = {'class_weight': weights}
        grid = GridSearchCV(estimator=model, param_grid=params, scoring='roc_auc', cv=k_fold, n_jobs=-1)
        grid.fit(X_train, y_train)
        print('best estimator for gridsearch on validation set: {}'.format(grid.best_score_))
        model = grid.best_estimator_
        probs_y = model.predict_proba(X_test)

        precision, recall, thresholds = precision_recall_curve(y_test, probs_y[:,1])
        pr_auc = metrics.auc(recall, precision)
        return precision, recall, thresholds, pr_auc, model

    def advance_logistic_model(self, X_train, X_test, y_train, y_test):

        model = LogisticRegression(random_state=13,
                                   class_weight='balanced',
                                   penalty='l1',
                                   fit_intercept=False,
                                   C=0.5,
                                   solver='liblinear')
        model.fit(X_train, y_train)
        probs_y = model.predict_proba(X_test)

        precision, recall, thresholds = precision_recall_curve(y_test, probs_y[:, 1])
        pr_auc = metrics.auc(recall, precision)
        return precision, recall, thresholds, pr_auc

    def balncer_cv(self, X, y, fold):
        kf = KFold(n_splits=fold)
        for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]
            sm = SMOTE()
            X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)
            model = LogisticRegression(random_state=13,
                                       class_weight='balanced',
                                       penalty='l1',
                                       fit_intercept=False,
                                       C=0.5,
                                       solver='liblinear')
            model.fit(X_train_oversampled, y_train_oversampled )
            y_pred = model.predict(X_test)
            print(f'For fold {fold}:')
            print(f'Accuracy: {model.score(X_test, y_test)}')
            print(f'f-score: {f1_score(y_test, y_pred)}')

    def score_model(self, X_train, y_train, model, cv=None):
        if cv is None:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

        smoter = SMOTE(random_state=42)

        scores = []

        for fold, (train_fold_index, val_fold_index) in enumerate(cv.split(X_train), 1):
            X_train_fold = X_train.iloc[train_fold_index]
            y_train_fold = y_train.iloc[train_fold_index]
            X_val_fold, y_val_fold = X_train.iloc[val_fold_index], y_train.iloc[val_fold_index]

            X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,
                                                                               y_train_fold)
            model_obj = model.fit(X_train_fold_upsample, y_train_fold_upsample)
            score = f1_score(y_val_fold, model_obj.predict(X_val_fold))
            scores.append(score)
        return np.array(scores)


if __name__ == '__main__':
    data_path = os.path.join(ROOT_PATH, 'data/')
    model_path = os.path.join(ROOT_PATH, 'model/')
    data = pd.read_csv(data_path + 'processed_data.csv')
    logistic_model = LogisticModel()
    X_train, X_test, y_train, y_test = logistic_model.data_splitting(data)
    X_train, X_test = logistic_model.feature_standardization(X_train, X_test)
    # X_train, y_train = logistic_model.over_smapling(X_train, y_train)
    # X_train, y_train = logistic_model.under_smapling(X_train, y_train)
    X_train, X_test = logistic_model.pca(X_train, X_test)
    # X_train, X_test = logistic_model.feature_selection(X_train, X_test)
    # logistic_model.simple_logistic_model(X_train, X_test, y_train, y_test)
    # logistic_model.weighted_logistic_model(X_train, X_test, y_train, y_test)
    # logistic_model.weight_grid_search(X_train, y_train)
    precision, recall, thresholds, pr_auc, model = logistic_model.model_weighted_grid(X_train, X_test, y_train, y_test)
    # scores = logistic_model.learning_curve(X_train, y_train)
    # plt.plot(range(1, 65), scores, linewidth=4)
    # plt.title("RandomUnderSampler Learning Curve", fontsize=16)
    # plt.gca().set_xlabel("# of Points per Class", fontsize=14)
    # plt.gca().set_ylabel("Training Accuracy", fontsize=14)
    # sns.despine()
    # plt.show()
    # X = data.drop('clicked', axis=1)
    # y = data['clicked']
    # logistic_model.balncer_cv(X, y, 5)
    # precision, recall, thresholds, pr_auc = logistic_model.advance_logistic_model(X_train, X_test, y_train, y_test)
    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0, 1])
    plt.show()
    print('AUC is: {}'.format(pr_auc))
    print('Precision is: {}'.format(precision))
    print('Recall is: {}'.format(recall))
    joblib_file = os.path.join(model_path, "LR_Model.pkl")
    joblib.dump(model, joblib_file)
    # model = LogisticRegression(random_state=13,
    #                            class_weight='balanced',
    #                            penalty='l1',
    #                            fit_intercept=False,
    #                            C=0.5,
    #                            solver='liblinear')
    # scores = logistic_model.score_model(X_train, y_train, model, cv=None)
    # print(scores)
