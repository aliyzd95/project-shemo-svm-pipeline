seed_value = 42
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)

import opensmile
from sklearn.multiclass import OneVsOneClassifier
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
import mlflow.sklearn
import yaml
import pickle
import pandas as pd
import dagshub

dagshub.init(repo_owner='aliyzd95', repo_name='project-shemo-svm-pipeline', mlflow=True)

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/aliyzd95/project-shemo-svm-pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "aliyzd95"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "5add05d8d42854133eb3f9fe9dcbb57b2360829d"

params = yaml.safe_load(open("params.yaml"))["train"]


def train(data_path, model_path, random_state):
    df_csv = pd.read_csv(data_path)

    features = []
    labels = []

    print("start feature extraction ... ")

    feature_extractor = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        verbose=True, num_workers=None,
        sampling_rate=16000, resample=True)

    for _, row in df_csv.iterrows():
        path = row['path']
        label = row['label']

        df_features = feature_extractor.process_file(f'{path}')
        features.append(df_features.values.squeeze())
        labels.append(label)

    X = np.array(features)
    y = np.array(labels)

    print("feature extraction done! ")

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    # mlflow.set_tracking_uri('http://127.0.0.1:5000')

    signature = mlflow.models.infer_signature(X, y)

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    model = SVC()
    ovo = OneVsOneClassifier(model)

    space = dict()
    space['estimator__C'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    space['estimator__gamma'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

    search = BayesSearchCV(ovo, space, scoring='recall_macro', cv=cv_inner, n_jobs=-1, verbose=0)
    pipeline = make_pipeline(StandardScaler(), search)

    with mlflow.start_run(run_name="SER-SVM-pipeline"):
        scores = {'test_accuracy': [], 'test_recall_macro': []}

        for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
            print(f"starting fold {fold_idx+1}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline.fit(X_train, y_train)

            best_params = pipeline.named_steps['bayessearchcv'].best_params_
            best_score = pipeline.named_steps['bayessearchcv'].best_score_
            best_model = pipeline.named_steps['bayessearchcv'].best_estimator_

            y_pred = pipeline.predict(X_test)
            acc = np.mean(y_pred == y_test)
            recall = recall_score(y_test, y_pred, average='macro')

            with mlflow.start_run(run_name=f"Fold_{fold_idx + 1}", nested=True):
                mlflow.log_params(best_params)
                mlflow.log_metric("best_inner_recall_score", best_score)
                mlflow.log_metric("best_outer_acc_score", acc)
                mlflow.log_metric("best_outer_recall_score", recall)
                mlflow.sklearn.log_model(best_model, f"best_model_fold_{fold_idx + 1}", signature=signature)

            scores['test_accuracy'].append(acc)
            scores['test_recall_macro'].append(recall)

            os.makedirs(model_path, exist_ok=True)
            pickle.dump(best_model, open(f"{model_path}SVM_Fold_{fold_idx + 1}.pkl", 'wb'))

            print(f"{fold_idx+1} saved!")

        print('____________________ Support Vector Machine ____________________')
        print(f"Weighted Accuracy: {np.mean(scores['test_accuracy']) * 100:.2f}")
        print(f"Unweighted Accuracy: {np.mean(scores['test_recall_macro']) * 100:.2f}")


if __name__ == "__main__":
    train(params['data'], params['model'], params["random_state"])
