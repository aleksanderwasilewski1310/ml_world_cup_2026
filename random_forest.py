"""Handles training, evaluation, and prediction using RandomForest for the FIFA dataset"""

import os
import pandas as pd
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


class FootballModelTrainer:
    def __init__(
        self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42
    ):
        """
        Initializes the FootballModelTrainer configuration.
        Uses MultiOutputClassifier to support simultaneous prediction of multiple tournament stages.
        """
        # Base Random Forest estimator with robust, generalized hyperparameters
        base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,  # Use all available CPU cores for speed
        )

        # Wrap the base classifier to handle multiple target columns simultaneously
        self.model = MultiOutputClassifier(base_rf)
        self.is_trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Trains the MultiOutput Random Forest model on the processed training features.
        """
        print(
            "Training Multi-Output RandomForestClassifier on historical World Cup data..."
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model training completed successfully.")
        return self

    def evaluate(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Evaluates the model on the training set (in-sample) since we don't have a separate validation set.
        Prints professional classification reports for each target stage.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")

        predictions = self.model.predict(X_train)
        predictions_df = pd.DataFrame(predictions, columns=y_train.columns)

        print("\n=== HISTORICAL TRAINING EVALUATION METRICS ===")
        for col in y_train.columns:
            print(f"\nTarget Stage: {col.upper()}")
            print(
                classification_report(
                    y_train[col], predictions_df[col], zero_division=0
                )
            )

    def predict_probabilities(
        self, X_test: pd.DataFrame, test_teams: pd.Series
    ) -> pd.DataFrame:
        """
        Predicts probabilities for unseen test data (World Cup 2026) and maps them to team names.
        Returns a beautifully structured DataFrame with forecasts.
        """
        if not self.is_trained:
            raise ValueError(
                "Model must be trained before making inference predictions."
            )

        print(
            "Calculating tournament progression probabilities for World Cup 2026 teams..."
        )

        # predict_proba returns a list of arrays (one array per target column)
        # Each array contains [prob_of_0, prob_of_1]
        proba_list = self.model.predict_proba(X_test)

        # Build a structured results DataFrame initialized with team names
        results_df = pd.DataFrame({"team": test_teams})

        # Iterate over each target output and extract the probability of success (class 1)
        for i, target_col in enumerate(
            X_train_columns_placeholder := self.model.estimators_
        ):
            # Target name can be inferred from the original order of training columns
            # proba_list[i][:, 1] gets the probability of getting a '1' (e.g. being a winner)
            pass

        # Hardcoded target iteration based on input design sequence for safety
        targets = ["winner", "finalist", "semi_finalist", "quarter_finalist"]
        for idx, target in enumerate(targets):
            results_df[f"{target}_probability"] = proba_list[idx][:, 1]

        return results_df


def train_and_predict_pipeline(
    processed_data: Dict[str, pd.DataFrame],
    raw_test_path: str = "./datasets/fifa/test.csv",
) -> pd.DataFrame:
    """
    Orchestration function that receives the processed data dictionary,
    extracts matrices, runs the training lifecycle, and produces 2026 forecasts.
    """
    # 1. Unpack data from the processing dictionary
    X_train = processed_data["X_train"]
    y_train = processed_data["y_train"]
    X_test = processed_data["X_test"]

    # 2. Extract original team names from the raw test file to align predictions
    if os.path.exists(raw_test_path):
        raw_test_df = pd.read_csv(raw_test_path)
        test_teams = raw_test_df["team"]
    else:
        # Fallback dummy naming if file path is unreachable
        test_teams = pd.Series([f"Team_{i}" for i in range(len(X_test))])
        print(
            f"Warning: Raw test file not found at {raw_test_path}. Using generic team names."
        )

    # 3. Initialize and execute model pipeline
    trainer = FootballModelTrainer(n_estimators=150, max_depth=8, random_state=42)
    trainer.train(X_train, y_train)

    # Run evaluation to verify historical calibration
    trainer.evaluate(X_train, y_train)

    # 4. Generate forecasts for 2026
    predictions_2026 = trainer.predict_probabilities(X_test, test_teams)

    # 5. Display a polished summary sorted by winner probability
    print("\n=== WORLD CUP 2026 PREDICTED FAVORITES ===")
    top_favorites = predictions_2026.sort_values(
        by="winner_probability", ascending=False
    )
    print(top_favorites.to_string(index=False))

    return predictions_2026


if __name__ == "__main__":
    # Integration simulation test
    # This acts as the link connecting your modules/data_processor.py output
    from modules.data_processor import data_processing

    try:
        # 1. Get the dictionary from your processor
        matrices = data_processing(data_folder="./datasets/fifa")

        # 2. Feed it into our new training pipeline function
        forecast_results = train_and_predict_pipeline(
            matrices, raw_test_path="./datasets/fifa/test.csv"
        )

    except (ModuleNotFoundError, FileNotFoundError) as e:
        print(
            f"Simulation setup incomplete. Ensure modules are in path and data is downloaded. Error: {e}"
        )
