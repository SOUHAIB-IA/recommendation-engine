# Movie Recommendation System - Backend

This repository contains the backend of the Movie Recommendation System, powered by Flask. The API interacts with a recommendation model to provide movie suggestions based on content-based filtering.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Setup](#setup)
- [API Endpoints](#api-endpoints)
- [Folder Structure](#folder-structure)
- [Deployment](#deployment)

### Overview

The backend application serves as the recommendation engine for the movie platform, processing user requests and returning relevant recommendations based on content similarity. It leverages Flask to expose endpoints that interact with a Supabase database and utilize cosine similarity calculations for recommendations.

### Features

- **Content-Based Filtering**: Uses cosine similarity to recommend movies based on user preferences.
- **Efficient Data Processing**: Preprocessed movie data, enriched with TMDB metadata.
- **Modular SOA Design**: Integrates seamlessly with frontend and database services.

### Technology Stack

- **Framework**: Flask for handling REST API requests
- **Database**: Supabase (PostgreSQL instance) for user and movie data storage
- **Modeling and Recommendations**: Cosine similarity calculations with Scikit-Learn
- **Deployment**: Hosted on Microsoft Azure for scalability and reliability

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/backend-movie-recommendation.git
   cd backend-movie-recommendation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** with necessary environment variables:

   ```plaintext
   FLASK_APP=app.py
   FLASK_ENV=development
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_connection_string
   ```

4. **Run the Flask server**:
   ```bash
   flask run
   ```

### API Endpoints

- **GET `/recommend`**: Accepts a POST request with a JSON payload containing the movie title. Returns a list of recommended movies.

  Example Request:
  ```json
  {
    "title": "Inception"
  }
  ```

  Example Response:
  ```json
  {
    "recommendations": ["The Matrix", "Interstellar", "Shutter Island"]
  }
  ```

### Folder Structure

- `app.py`: Main file to start the Flask app and define endpoints.
- `utils/`: Utility functions, including data processing and similarity calculation.
- `models/`: Contains model files for recommendation calculations.
- `artifacts/`: Stores precomputed similarity matrices and other data artifacts.

### Deployment

1. **GitHub Actions Workflow**: Automate deployment to Azure Web App.
2. **Configuration**: Modify `.env` and `requirements.txt` as needed for your Azure environment.
3. **Azure Setup**: Upload code to Azure and run the API on a web server like Gunicorn.
