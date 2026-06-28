import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


def engineer_features(train, test):
    # Total living area
    for df in [train, test]:
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
        df["TotalBathrooms"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

    binary_features = {
        "HasGarage": "GarageArea",
        "HasBasement": "TotalBsmtSF",
        "HasFireplace": "Fireplaces",
        "HasPool": "PoolArea",
        "HasSecondFloor": "2ndFlrSF",
    }
    for new_feature, feature in binary_features.items():
        for df in [train, test]:
            df[new_feature] = (df[feature] > 0).astype(int)

    log_features = ["GrLivArea", "LotArea", "TotalBsmtSF", "1stFlrSF"]
    for feature in log_features:
        for df in [train, test]:
            df[feature] = np.log1p(df[feature])

    train["SalePrice"] = np.log1p(train["SalePrice"])

    outlier_mask = ~(
        (train["GrLivArea"] > np.log1p(4000)) & (train["SalePrice"] < np.log1p(300000))
    )
    train = train.loc[outlier_mask].copy()

    return train, test


def build_preprocessor(X_train):
    ordinal_features = [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "Functional",
        "GarageFinish",
        "Fence",
        "LotShape",
        "LandSlope",
        "PavedDrive",
        "Utilities",
    ]

    ordinal_categories = [
        ["Po", "Fa", "TA", "Gd", "Ex"],
        ["Po", "Fa", "TA", "Gd", "Ex"],
        ["None", "Po", "Fa", "TA", "Gd", "Ex"],
        ["None", "Po", "Fa", "TA", "Gd", "Ex"],
        ["Po", "Fa", "TA", "Gd", "Ex"],
        ["Po", "Fa", "TA", "Gd", "Ex"],
        ["None", "Po", "Fa", "TA", "Gd", "Ex"],
        ["None", "Po", "Fa", "TA", "Gd", "Ex"],
        ["None", "Po", "Fa", "TA", "Gd", "Ex"],
        ["None", "Fa", "TA", "Gd", "Ex"],
        ["None", "No", "Mn", "Av", "Gd"],
        ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
        ["None", "Unf", "RFn", "Fin"],
        ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
        ["IR3", "IR2", "IR1", "Reg"],
        ["Sev", "Mod", "Gtl"],
        ["N", "P", "Y"],
        ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    ]

    numerical_features = X_train.select_dtypes(include="number").columns

    nominal_features = [
        col
        for col in X_train.select_dtypes(include="str").columns
        if col not in ordinal_features
    ]

    numerical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    nominal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numerical_pipeline, numerical_features),
            ("ord", ordinal_pipeline, ordinal_features),
            ("nom", nominal_pipeline, nominal_features),
        ]
    )

    return preprocessor
