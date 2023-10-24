# saved_xgb_regression_model.py

import numpy as np
from xgboost import XGBRegressor
from scipy.stats import uniform


class OptimizedXGBRegressor:
    def __init__(self, params=None):
        self.params = params or {
            'learning_rate': uniform(0.01, 0.3),
            'n_estimators': np.arange(50, 300, 10),
            'max_depth': np.arange(3, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 1),
        }
        self.regressor = XGBRegressor()

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)

    def get_params(self):
        return self.regressor.get_params()

    def set_params(self, **params):
        self.regressor.set_params(**params)


if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import RandomizedSearchCV

    data = load_boston()
    X, y = data.data, data.target

    optimizer = OptimizedXGBRegressor()


    def custom_scorer(estimator, X, y):
        return np.mean(cross_val_score(estimator, X, y, cv=5))


    scorer = make_scorer(custom_scorer, greater_is_better=True)

    random_search = RandomizedSearchCV(
        optimizer,
        param_distributions=optimizer.params,
        n_iter=20,
        scoring=scorer,
        n_jobs=-1,
        verbose=1,
        cv=5,
    )

    random_search.fit(X, y)

    print("Best parameters:", random_search.best_params_)
    print("Best score:", random_search.best_score_)
