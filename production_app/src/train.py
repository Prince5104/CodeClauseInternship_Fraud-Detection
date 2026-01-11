from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X_train, y_train):
    params ={
        "n_estimators": [200],
        "max_depth": [10],
        "learning_rate": [0.1]
    }

    model = XGBClassifier(eval_metric="logloss")
    grid = GridSearchCV(model, params, scoring="roc_auc", cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def save_model(model, path="models/fraud_xgb.pkl"):
    joblib.dump(model, path)