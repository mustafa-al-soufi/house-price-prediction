import numpy as np
import pandas as pd
from train import train


def predict(output_path="../data/submission.csv"):
    final_lasso, preprocessor, test = train()

    X_test_prepared = pd.DataFrame(
        preprocessor.transform(test),
        columns=preprocessor.get_feature_names_out(),
        index=test.index,
    )

    predictions = np.expm1(final_lasso.predict(X_test_prepared))

    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": predictions})

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(submission.head())


if __name__ == "__main__":
    predict()
