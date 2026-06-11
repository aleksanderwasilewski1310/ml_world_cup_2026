"""FastAPI application providing endpoint for FIFA World Cup 2026 predictions"""

# ruff: noqa: E402
import pandas as pd
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Critical: Load env variables before custom imports to satisfy Kaggle API authentication sequence
load_dotenv()

from modules.data_loader import DataSet
from modules.data_processor import data_processing
from random_forest import train_and_predict_pipeline

# 1. Global state container to store prediction results in RAM
app_data = {}


# 2. Define Pydantic schema for robust API response validation and auto-generated docs
class ProbabilityResponse(BaseModel):
    team: str
    winner_probability: str
    finalist_probability: str
    semi_finalist_probability: str
    quarter_finalist_probability: str


def show_probabilities(country: str, forecast_data: pd.DataFrame) -> dict:
    """
    Filters and formats the forecast dataframe for a specific country.
    """
    country_cleaned = country.strip().title()

    if country_cleaned not in forecast_data["team"].values:
        raise HTTPException(
            status_code=404,
            detail=f"Country '{country}' does not play in FIFA World Cup 2026 or is misspelled.",
        )

    country_row = forecast_data[forecast_data["team"] == country_cleaned].iloc[0]

    return {
        "team": str(country_row["team"]),
        "winner_probability": f"{float(country_row['winner_probability']) * 100:.2f}%",
        "finalist_probability": f"{float(country_row['finalist_probability']) * 100:.2f}%",
        "semi_finalist_probability": f"{float(country_row['semi_finalist_probability']) * 100:.2f}%",
        "quarter_finalist_probability": f"{float(country_row['quarter_finalist_probability']) * 100:.2f}%",
    }


# 3. Modern FastAPI Lifespan context manager to handle pipeline trigger on server startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[STARTUP] Executing Machine Learning Pipeline...")
    try:
        # Load and download raw data if missing
        data_set = DataSet()
        data_set.load_data()

        # Run preprocessing
        matrices = data_processing(data_folder="./datasets/fifa")

        # Train RandomForest and score probabilities for 2026
        forecast_results = train_and_predict_pipeline(
            matrices, raw_test_path="./datasets/fifa/test.csv"
        )

        # Cache the resulting DataFrame globally in RAM
        app_data["forecast_results"] = forecast_results
        print("[STARTUP] ML Model ready. Application initialized successfully.\n")
    except Exception as e:
        print(f"[STARTUP ERROR] Pipeline failed to initialize: {e}")
        # We don't crash the server, but subsequent API calls will notice missing state
        app_data["forecast_results"] = None

    yield
    print("[SHUTDOWN] Cleaning up server resources...")


# Initialize FastAPI app with metadata and lifespan orchestration
app = FastAPI(
    title="FIFA World Cup 2026 Prediction API",
    description="Machine Learning service delivering tournament progression forecasts using Random Forest.",
    version="1.0.0",
    lifespan=lifespan,
)


# 4. Define the prediction endpoint
@app.get(
    "/api/v1/predictions",
    response_model=ProbabilityResponse,
    summary="Get probabilities by country name",
)
async def get_team_probabilities(
    country: str = Query(
        ..., description="Name of the country, e.g. 'France', 'Czech Republic'"
    ),
):
    """
    Fetches processed tournament progression probabilities from the cached model results.
    """
    forecast_df = app_data.get("forecast_results")

    if forecast_df is None:
        raise HTTPException(
            status_code=503,
            detail="Model predictions are unavailable. Check server logs for initialization errors.",
        )

    # Delegate formatting logic to our clean show_probabilities helper
    result = show_probabilities(country, forecast_df)
    return result
