import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np


def evaluate_model(model, X_train, y_train, X_valid, y_valid, model_name="Model"):
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    train_rmse = root_mean_squared_error(y_train, train_pred)
    valid_rmse = root_mean_squared_error(y_valid, valid_pred)
    train_r2 = r2_score(y_train, train_pred)
    valid_r2 = r2_score(y_valid, valid_pred)

    print(f"{model_name}")
    print(f"  Train RMSE:      {train_rmse:.6f}")
    print(f"  Validation RMSE: {valid_rmse:.6f}")
    print(f"  Train R²:        {train_r2:.6f}")
    print(f"  Validation R²:   {valid_r2:.6f}")

    return {
        "Model": model_name,
        "Train RMSE": train_rmse,
        "Validation RMSE": valid_rmse,
        "Train R²": train_r2,
        "Validation R²": valid_r2,
    }


def cross_validate_model(model, X, y, model_name="Model", cv=5):
    scores = cross_val_score(
        model, X, y, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1
    )
    mean_rmse = -scores.mean()
    std_rmse = scores.std()

    print(f"{model_name} — Mean CV RMSE: {mean_rmse:.6f} ± {std_rmse:.6f}")

    return {"Model": model_name, "Mean RMSE": mean_rmse, "Standard Deviation": std_rmse}
