from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from azure.storage.blob import BlobServiceClient

# Initialize Flask application
app = Flask(__name__)


# Get Azure Blob Storage connection string and container name from environment variables
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

# Ensure the environment variables are set
if not AZURE_STORAGE_CONNECTION_STRING or not CONTAINER_NAME:
    raise ValueError("Azure Storage connection string or container name not set in environment variables.")

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

def download_blob(blob_name, download_path):
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        with open(download_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
    except Exception as e:
        print(f"Failed to download blob {blob_name}: {e}")
        raise

# Paths where the joblib files will be downloaded/stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movie_list_path = os.path.join(BASE_DIR, 'artifacts', 'movie_list.joblib')
similarity_matrix_path = os.path.join(BASE_DIR, 'artifacts', 'similarity.joblib')

# Download the joblib files if they don't exist locally
if not os.path.exists(movie_list_path):
    download_blob("movie_list.joblib", movie_list_path)

if not os.path.exists(similarity_matrix_path):
    download_blob("similarity.joblib", similarity_matrix_path)

# Load the joblib files
try:
    movies = joblib.load(movie_list_path)
    similarity_matrix = joblib.load(similarity_matrix_path)
except Exception as e:
    print(f"Error loading model files: {e}")
    raise


@app.route('/')
def home():
    return "Welcome to the Movie Recommendation API!"

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Recommends similar movies based on the provided movie title.
    
    Expects JSON input: {"title": "movie_name"}
    """
    data = request.get_json()
    movie_title = data.get('title')

    # Check if the movie title is provided in the request
    if not movie_title:
        return jsonify({"error": "Movie title not provided."}), 400

    # Find the movie index based on the provided title
    try:
        # Perform case-insensitive search for the movie title in the dataset
        movie_index = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
        movie_index = int(movie_index)  # Convert the index to a native Python integer type
    except IndexError:
        return jsonify({"error": f"Movie '{movie_title}' not found in the dataset."}), 404
    except Exception as e:
        print(f"An unexpected error occurred while searching for the movie: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

    print(f"Movie Index: {movie_index} (Type: {type(movie_index)})")

    try:
        # Validate similarity matrix
        if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            return jsonify({"error": "Invalid similarity matrix shape. It must be square."}), 500

        # Retrieve similarity scores for the given movie index
        similarity_scores = list(enumerate(similarity_matrix[movie_index]))
        print(f"Similarity Scores: {similarity_scores[:5]}")  # Print first 5 similarity scores for debugging

        # Sort the similarity scores in descending order and get the top 10 recommendations (excluding the first)
        sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]
        print(f"Sorted Similar Movies: {sorted_similar_movies}")

        # Get the movie titles of the top recommendations
        recommended_titles = [movies.iloc[i[0]].title for i in sorted_similar_movies]

        return jsonify({"recommendations": recommended_titles})

    except Exception as e:
        print(f"An error occurred while processing the similarity matrix: {e}")
        return jsonify({"error": "An error occurred while processing the similarity matrix."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))