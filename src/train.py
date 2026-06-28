import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import root_mean_squared_error, r2_score
from preprocessing import engineer_features, build_preprocessor


def train(train_path="../data/train.csv", test_path="../data/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, test = engineer_features(train, test)

    X = train.drop("SalePrice", axis=1)
    y = train["SalePrice"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    X_train_prepared = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=preprocessor.get_feature_names_out(),
        index=X_train.index,
    )
    X_valid_prepared = pd.DataFrame(
        preprocessor.transform(X_valid),
        columns=preprocessor.get_feature_names_out(),
        index=X_valid.index,
    )

    # Randomized search
    lasso_search = RandomizedSearchCV(
        Lasso(max_iter=10000),
        {"alpha": np.logspace(-4, 1, 100)},
        n_iter=50,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    lasso_search.fit(X_train_prepared, y_train)

    # Grid search around best alpha
    best_alpha = lasso_search.best_params_["alpha"]
    lasso_grid_search = GridSearchCV(
        Lasso(max_iter=10000),
        {"alpha": np.logspace(np.log10(best_alpha) - 1, np.log10(best_alpha) + 1, 50)},
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    lasso_grid_search.fit(X_train_prepared, y_train)

    print("Best alpha:", lasso_grid_search.best_params_)
    print("Best CV RMSE:", -lasso_grid_search.best_score_)

    # Train final model on full data
    X_full = np.vstack([X_train_prepared, X_valid_prepared])
    y_full = pd.concat([y_train, y_valid]).reset_index(drop=True)

    final_lasso = Lasso(alpha=lasso_grid_search.best_params_["alpha"], max_iter=10000)
    final_lasso.fit(X_full, y_full)

    return final_lasso, preprocessor, test


if __name__ == "__main__":
    train()
