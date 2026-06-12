import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Assuming your original code is saved in a file named: model_trainer.py
from random_forest import FootballModelTrainer, train_and_predict_pipeline


# =====================================================================
# FIXTURES - Generation of synthetic data for isolated testing
# =====================================================================


@pytest.fixture
def dummy_train_data():
    """
    Generates a reproducible, stable synthetic training dataset.
    Creates 20 rows of data with 5 independent features and 4 target stages.
    """
    np.random.seed(42)
    # Generate 20 rows and 5 features
    X = pd.DataFrame(np.random.randn(20, 5), columns=[f"feature_{i}" for i in range(5)])

    # 4 tournament stages as binary targets (0 or 1)
    targets = ["winner", "finalist", "semi_finalist", "quarter_finalist"]
    y = pd.DataFrame(np.random.randint(0, 2, size=(20, 4)), columns=targets)
    return X, y


@pytest.fixture
def dummy_test_data():
    """
    Generates synthetic unseen inference test data for World Cup 2026 simulation.
    Creates 3 test instances corresponding to 3 national teams.
    """
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(3, 5), columns=[f"feature_{i}" for i in range(5)])
    teams = pd.Series(["Poland", "Sweden", "Brazil"])
    return X, teams


# =====================================================================
# UNIT TESTS - FootballModelTrainer Lifecycle and Edge Cases
# =====================================================================


def test_trainer_initialization():
    """
    Verifies that the MultiOutputClassifier wrapper and its underlying
    Base RandomForest hyperparameters initialize with correct configurations.
    """
    trainer = FootballModelTrainer(n_estimators=50, max_depth=5, random_state=10)

    assert trainer.is_trained is False
    assert trainer.model.estimator.n_estimators == 50
    assert trainer.model.estimator.max_depth == 5
    assert trainer.model.estimator.random_state == 10


def test_error_raised_when_evaluating_untrained_model(dummy_train_data):
    """
    Ensures that a ValueError with an appropriate message is raised
    if evaluation is triggered before invoking the fit/train process.
    """
    X, y = dummy_train_data
    trainer = FootballModelTrainer()

    with pytest.raises(ValueError, match="Model must be trained before evaluation."):
        trainer.evaluate(X, y)


def test_error_raised_when_predicting_untrained_model(dummy_test_data):
    """
    Ensures that a ValueError with an appropriate message is raised
    if inference/prediction is triggered before invoking the fit/train process.
    """
    X, teams = dummy_test_data
    trainer = FootballModelTrainer()

    with pytest.raises(
        ValueError, match="Model must be trained before making inference predictions."
    ):
        trainer.predict_probabilities(X, teams)


def test_model_training_lifecycle(dummy_train_data, dummy_test_data):
    """
    End-to-End Unit Test verifying successful training execution,
    in-sample evaluation stability, and correctness of output probability shapes.
    """
    X_train, y_train = dummy_train_data
    X_test, test_teams = dummy_test_data

    trainer = FootballModelTrainer(n_estimators=10, max_depth=3, random_state=42)

    # 1. Test Training execution & Fluent API interface
    returned_trainer = trainer.train(X_train, y_train)
    assert trainer.is_trained is True
    assert returned_trainer == trainer  # Verifies 'return self' behavior

    # 2. Test Evaluation stability (ensuring classification_report generation does not crash)
    try:
        trainer.evaluate(X_train, y_train)
    except Exception as e:
        pytest.fail(f"Method evaluate() raised an unexpected exception: {e}")

    # 3. Test Prediction and Inference Dataframe properties
    results = trainer.predict_probabilities(X_test, test_teams)

    assert isinstance(results, pd.DataFrame), "Output must be a pandas DataFrame"
    assert "team" in results.columns, "Output DataFrame missing 'team' mapping column"
    assert len(results) == len(
        test_teams
    ), "Row dimension mismatch between input teams and output forecasts"

    # Verify probability formatting and mathematical bound constraints
    expected_targets = ["winner", "finalist", "semi_finalist", "quarter_finalist"]
    for target in expected_targets:
        col_name = f"{target}_probability"
        assert (
            col_name in results.columns
        ), f"Expected output probability column '{col_name}' missing"
        # Probabilities must mathematically fall within the [0.0, 1.0] interval
        assert (
            results[col_name].between(0.0, 1.0).all()
        ), f"Probabilities in {col_name} out of bounds"


# =====================================================================
# INTEGRATION TESTS - Orchestration Pipeline Behavior & I/O Mocking
# =====================================================================


@patch("os.path.exists")
def test_pipeline_fallback_when_test_file_missing(
    mock_exists, dummy_train_data, dummy_test_data
):
    """
    Validates the orchestration pipeline's robust fallback mechanism.
    When the required raw test CSV file is missing from the environment,
    the system must seamlessly generate generic placeholder names.
    """
    X_train, y_train = dummy_train_data
    X_test, _ = dummy_test_data

    # Simulate that the file path does NOT exist on the file system
    mock_exists.return_value = False

    processed_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test}

    # Execute orchestration pipeline using an unreachable path
    predictions = train_and_predict_pipeline(
        processed_data, raw_test_path="invalid_path.csv"
    )

    # Verify execution of fallback naming pattern (Team_0, Team_1, etc.)
    assert (
        predictions["team"].iloc[0] == "Team_0"
    ), "Fallback naming convention failed to execute"
    assert predictions["team"].iloc[1] == "Team_1"
    assert len(predictions) == len(X_test)
