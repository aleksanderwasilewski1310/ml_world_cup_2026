"""Handles data preprocessing, cleaning, and feature engineering for the FIFA dataset"""

import pandas as pd
import os
from typing import Tuple, List, Dict


class DataProcessor:
    def __init__(self, target_columns: List[str] = None):
        """
        Initializes the DataProcessor configuration.
        """
        if target_columns is None:
            self.target_columns = [
                "winner",
                "finalist",
                "semi_finalist",
                "quarter_finalist",
            ]
        else:
            self.target_columns = target_columns

        # Placeholders for fitted transformers or statistical values
        self.imputation_values: Dict = {}
        self.feature_columns: List[
            str
        ] = []  # CRITICAL: Stores exact training feature schema
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        """
        Fits the preprocessor using the training dataset.
        Calculates and stores statistics like medians for robust future imputation.
        """
        df = train_df.copy()

        # Calculate median market values grouped by continent and tournament version
        self.imputation_values["market_value_fallback"] = df[
            "squad_total_market_value_eur"
        ].median()
        self.imputation_values["grouped_market_medians"] = (
            df.groupby(["version", "continent"])["squad_total_market_value_eur"]
            .median()
            .to_dict()
        )

        return self

    def transform(
        self, input_df: pd.DataFrame, drop_year_2002: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transforms the input DataFrame by handling missing values, encoding categories,
        and separating features from targets. Enforces strict schema consistency.
        """
        df = input_df.copy()

        # 1. Handle Missing Data (The 2002 Market Value Challenge)
        if drop_year_2002 and 2002 in df["version"].values:
            print(
                "Filtering out historical data from year 2002 due to systemic missing financial values."
            )
            df = df[df["version"] != 2002].reset_index(drop=True)
        else:
            df["squad_total_market_value_eur"] = df.apply(
                lambda row: self._impute_market_value(row), axis=1
            )

        # 2. Feature Engineering: Create derived domain-specific features
        df["value_per_age_ratio"] = df["squad_total_market_value_eur"] / (
            df["squad_avg_age"] + 1e-5
        )
        df["group_pass_efficiency"] = df["groups_passed_before"] / (
            df["world_cup_participations_before"] + 1e-5
        )
        df["goal_differential_last_4y"] = (
            df["goals_scored_last_4y"] - df["goals_received_last_4y"]
        )
        df["win_ratio_last_4y"] = df["wins_last_4y"] / (
            df["wins_last_4y"] + df["losses_last_4y"] + df["draws_last_4y"] + 1e-5
        )

        # 3. Categorical Encoding (One-Hot Encoding for Continent)
        df = pd.get_dummies(df, columns=["continent"], drop_first=False, dtype=int)

        # 4. Separate Features (X) and Targets (y)
        existing_targets = [col for col in self.target_columns if col in df.columns]

        if existing_targets:
            y = df[existing_targets]
            drop_cols = self.target_columns + ["team"]
            X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        else:
            y = pd.DataFrame()  # Empty DataFrame for inference/test scenarios
            X = df.drop(columns=["team"] if "team" in df.columns else [])

        # 5. CRITICAL SCHEMA ALIGNMENT (Prevents features mismatch between train and test)
        if not self.is_fitted:
            # If fitting, save the final column structure of the training set
            self.feature_columns = list(X.columns)
            self.is_fitted = True
            print("DataProcessor successfully fitted with training statistics.")
        else:
            # If transforming test data, align columns with the saved training schema
            # Reindex adds missing columns as NaN/0 and drops unexpected columns
            X = X.reindex(columns=self.feature_columns, fill_value=0)

        return X, y

    def fit_transform(
        self, train_df: pd.DataFrame, drop_year_2002: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Combines fit and transform steps in a single production-grade pipeline call.
        """
        return self.fit(train_df).transform(train_df, drop_year_2002=drop_year_2002)

    def _impute_market_value(self, row: pd.Series) -> float:
        """
        Internal helper for context-aware missing value imputation.
        """
        if not pd.isna(row["squad_total_market_value_eur"]):
            return row["squad_total_market_value_eur"]

        key = (row["version"], row["continent"])
        grouped_medians = self.imputation_values.get("grouped_market_medians", {})

        if key in grouped_medians and not pd.isna(grouped_medians[key]):
            return grouped_medians[key]

        return self.imputation_values["market_value_fallback"]


def data_processing(data_folder: str) -> dict:
    """Initializes data processing orchestration pipeline"""
    train_data_path = os.path.join(data_folder, "train.csv")
    test_data_path = os.path.join(data_folder, "test.csv")

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    processor = DataProcessor()

    # Process training data (fitting parameters and transforming)
    X_train, y_train = processor.fit_transform(train_data, drop_year_2002=True)
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Training targets shape: {y_train.shape}")

    # Process unseen test data aligned strictly with training features schema
    X_test, _ = processor.transform(test_data, drop_year_2002=False)
    print(f"Test features shape: {X_test.shape}")

    print("\nEngineered features preview (First 2 rows of X_train):")
    print(
        X_train[
            [
                "value_per_age_ratio",
                "group_pass_efficiency",
                "goal_differential_last_4y",
            ]
        ].head(2)
    )

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test}


if __name__ == "__main__":
    # Assuming datasets are stored in './datasets/fifa' directory
    processed_data = data_processing(data_folder="./datasets/fifa")
    print(processed_data)
