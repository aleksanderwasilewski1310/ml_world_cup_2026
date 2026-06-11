# 🏆 FIFA World Cup 2026 ML Predictor API

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning system that predicts the outcomes of the 2026 FIFA World Cup. The project automatically fetches historical data from Kaggle, processes it (Feature Engineering), trains Random Forest models, and exposes the predictions via a high-performance REST API built with FastAPI.

The entire application is fully containerized, secure (non-root execution), and production-ready.

## ✨ Key Features

* **Automated Data Pipeline:** Seamlessly downloads and extracts Kaggle datasets (`kaggle API v2`) on startup.
* **Advanced Feature Engineering:** Smart imputation for missing historical data (e.g., 2002 market values), domain-specific metrics creation (squad value-to-age ratio), and strict data leakage prevention.
* **Multi-label Machine Learning:** Utilizes a `MultiOutputClassifier` wrapped around a `RandomForestClassifier` to simultaneously predict multiple tournament stages: Winner, Finalist, Semi-finalist, and Quarter-finalist.
* **REST API:** Blazing-fast inference via FastAPI with automatic request validation (Pydantic) and auto-generated Swagger UI.
* **Docker & uv:** Extremely fast container builds using the `uv` package manager, configured with a secure non-root environment.

## 📂 Project Structure

```text
world_cup_2026/
├── datasets/
│   └── fifa/                 # Downloaded Kaggle CSVs (train.csv, test.csv)
├── docker/
│   ├── Dockerfile            # Multi-stage container definition using 'uv'
├── modules/
│   ├── data_loader.py        # Kaggle API authentication and data fetching
│   ├── data_processor.py     # Data cleaning, imputation, and feature engineering
│   └── model_trainer.py      # Random Forest training and prediction logic
├── .env                      # Environment variables (Kaggle API Token)
├── docker-compose.yml        # Docker services configuration
├── main.py                   # FastAPI Application Entrypoint
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
## 🚀 Quick Start (Docker - Recommended)

The easiest way to run the project is by using Docker.

### 1. Prerequisites
* **Docker** and **Docker Compose** installed.
* A registered account on [Kaggle](https://www.kaggle.com/).

### 2. Kaggle API Setup
The project requires a Kaggle API token to fetch historical World Cup data.
1. Log in to Kaggle, go to *Settings* -> *API* -> **Create New Token**.
2. Copy the generated token string.
3. In the root directory of the project, create a `.env` file and paste your token:

```env
# .env file
KAGGLE_API_TOKEN=your_generated_token_here
```
3. Run the Application
Build and start the container with a single command:

```Bash
docker compose up --build
```
The container will automatically set up secure user permissions, download the data, train the ML model, and spin up the server on port 8000.

💻 Local Setup (Without Docker)
If you prefer to run the project directly on your host machine:

# 1. Create and activate a virtual environment
```
python -m venv .venv
```

# On Windows:
```
.venv\Scripts\activate
```

# On Linux/Mac:
```
source .venv/bin/activate
```

# 2. Install dependencies
```
pip install -r requirements.txt
```

# 3. Ensure your .env file with KAGGLE_API_TOKEN is present

# 4. Start the API server
```
uvicorn main:app --reload
```

# 📡 API Documentation & Usage
Once the server successfully starts (you'll see Application initialized successfully in the logs), the API will be available at:
```
http://127.0.0.1:8000
```

Interactive Documentation (Swagger UI)
Open your browser and navigate to:
👉 http://127.0.0.1:8000/docs

Example Request (Endpoint)
```
GET /api/v1/predictions?country=Country Name

cURL Example:

Bash
curl -X 'GET' \
  '[http://127.0.0.1:8000/api/v1/predictions?country=France](http://127.0.0.1:8000/api/v1/predictions?country=France)' \
  -H 'accept: application/json'
```

JSON Response:

```JSON
{
  "team": "France",
  "winner_probability": "24.00%",
  "finalist_probability": "38.50%",
  "semi_finalist_probability": "55.00%",
  "quarter_finalist_probability": "72.00%"
}
```
# 🧪 Testing
The project includes unit tests using pytest, featuring comprehensive API mocking to prevent unnecessary network calls to Kaggle during testing.

To run the test suite locally:

```Bash
python -m pytest -v
```

# 🛠 Technology Stack
Language: Python 3.13

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn (Random Forest)

API Framework: FastAPI, Uvicorn, Pydantic

DevOps: Docker, Docker Compose, Astral 'uv'
