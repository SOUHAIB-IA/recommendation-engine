from flask import Flask, request, jsonify,Response
import threading
import joblib
import pandas as pd
import json
import numpy as np
import os
from azure.storage.blob import BlobServiceClient
import time

app = Flask(__name__)


AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")


if not AZURE_STORAGE_CONNECTION_STRING or not CONTAINER_NAME:
    raise ValueError("Azure Storage connection string or container name not set in environment variables.")


blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

def download_blob(blob_name, download_path):
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        with open(download_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
    except Exception as e:
        print(f"Failed to download blob {blob_name}: {e}")
        raise


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

artifacts_dir = os.path.join(BASE_DIR, 'artifacts')
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)
    
movie_list_path = os.path.join(BASE_DIR, 'artifacts', 'movie_list.joblib')
similarity_matrix_path = os.path.join(BASE_DIR, 'artifacts', 'similarity.joblib')
best_gbr_path= os.path.join(BASE_DIR, 'artifacts', 'temperature_model.joblib')
initial_sequenceT_path= os.path.join(BASE_DIR, 'artifacts', 'initial_sequenceT.joblib')
scaler_path=os.path.join(BASE_DIR, 'artifacts', 'scaler.joblib')
humidity_model_path = os.path.join(BASE_DIR, 'artifacts', 'humidity_model.joblib')
initial_sequenceH_path= os.path.join(BASE_DIR, 'artifacts', 'initial_sequenceH.joblib')
condec_model_path=os.path.join(BASE_DIR,'artifacts', 'condic_model.joblib')
initial_sequenceD_path= os.path.join(BASE_DIR, 'artifacts', 'initial_sequenceD.joblib')

if not os.path.exists(condec_model_path):
    download_blob('condic_model.joblib',condec_model_path)
if not os.path.exists(initial_sequenceD_path):
    download_blob('initial_sequenceD.joblib',initial_sequenceD_path)

if not os.path.exists(best_gbr_path):
    download_blob("temperature_model.joblib", best_gbr_path)

if not os.path.exists(initial_sequenceT_path):
    download_blob("initial_sequenceT.joblib", initial_sequenceT_path)

if not os.path.exists(scaler_path):
    download_blob("scaler.joblib",scaler_path)


if not os.path.exists(movie_list_path):
    download_blob("movie_list.joblib", movie_list_path)

if not os.path.exists(similarity_matrix_path):
    download_blob("similarity.joblib", similarity_matrix_path)

if not os.path.exists(humidity_model_path):
    download_blob("humidity_model.joblib", humidity_model_path)
try:
    movies = joblib.load(movie_list_path)
    similarity_matrix = joblib.load(similarity_matrix_path)
    best_gbr = joblib.load(best_gbr_path)
    initial_sequenceT= joblib.load(initial_sequenceT_path)
    scaler=joblib.load(scaler_path)
    humidity_model = joblib.load(humidity_model_path)
    initial_sequenceH = joblib.load(initial_sequenceH_path)
    condic_model=joblib.load(condec_model_path)
    initial_sequenceD=joblib.load(initial_sequenceD_path)

except Exception as e:
    print(f"Error loading model files: {e}")
    raise

simulated_data = []
humidity_data = []
condictive_data=[]
data_lock = threading.Lock()

def simulate_temperature_sensor(model, initial_sequence, scaler, interval=540, steps=100):
    """
    Simulates a real-time temperature sensor using a trained model with added realism.
    """

    sequence = np.array(initial_sequence)

    for _ in range(steps):
        next_scaled_value = model.predict(sequence.reshape(1, -1))
        next_temperature = scaler.inverse_transform(next_scaled_value.reshape(-1, 1))[0, 0]
        noise = np.random.normal(0, 0.5)
        next_temperature += noise
    
        seasonal_pattern = np.sin(time.time() / 86400 * 2 * np.pi) * 2  
        next_temperature += seasonal_pattern

      
        drift = 0.01  
        next_temperature += drift

        next_temperature = max(min(next_temperature, 40), -10) 

      
        if np.random.rand() < 0.1:
            next_temperature += np.random.choice([-5, 5])

        with data_lock:
            simulated_data.append(next_temperature)
        sequence = np.roll(sequence, -1) 
        sequence[-1] = next_scaled_value[0] 

       
        interval_variation = np.random.uniform(0.9, 1.1) 
        time.sleep(interval * interval_variation)

def simulate_soil_humidity_sensor(model, initial_sequence, scaler, interval=540, steps=100):
    sequence = np.array(initial_sequence)

    for _ in range(steps):
        next_scaled_value = model.predict(sequence.reshape(1, -1))
        next_humidity = scaler.inverse_transform(next_scaled_value.reshape(-1, 1))[0, 0]

        noise = np.random.normal(0, 1)
        next_humidity += noise

        seasonal_pattern = np.sin(time.time() / 86400 * 2 * np.pi) * 1.5
        next_humidity += seasonal_pattern

        drift = 0.005
        next_humidity += drift

        next_humidity = max(min(next_humidity, 100), 0)

        if np.random.rand() < 0.1:
            next_humidity += np.random.choice([-10, 10])

        with data_lock:
            humidity_data.append(next_humidity)

        sequence = np.roll(sequence, -1)
        sequence[-1] = next_scaled_value[0]

        interval_variation = np.random.uniform(0.9, 1.1)
        time.sleep(interval * interval_variation)

def simulate_electrical_conductivity_sensor(model, initial_sequence, scaler, interval=540, steps=100):
    """
    Simulates a real-time electrical conductivity sensor using a trained model with added realism.
    
    :param model: Trained GradientBoostingRegressor model
    :param initial_sequence: The starting sequence (last known electrical conductivity values)
    :param scaler: The scaler used for normalization
    :param interval: Time interval between sensor readings in seconds (default: 3 seconds)
    :param steps: Number of steps to simulate (default: 100)
    
    :return: None
    """
    sequence = np.array(initial_sequence)

    for _ in range(steps):
        # Predict the next value (scaled)
        next_scaled_value = model.predict(sequence.reshape(1, -1))
        
        # Inverse transform to get the predicted conductivity value
        next_conductivity = scaler.inverse_transform(next_scaled_value.reshape(-1, 1))[0, 0]
        
        # Add Gaussian noise to simulate sensor imperfections (mean=0, std=0.01)
        noise = np.random.normal(0, 0.01)
        next_conductivity += noise
        
        # Simulate seasonal fluctuations (small amplitude sine wave)
        seasonal_pattern = np.sin(time.time() / 86400 * 2 * np.pi) * 0.1  # Adding a small seasonal fluctuation
        next_conductivity += seasonal_pattern
        
        # Simulate drift (slow gradual change in conductivity)
        drift = 0.001  # Slight drift over time
        next_conductivity += drift
        
        # Enforce realistic conductivity range (0 to 10 S/m)
        next_conductivity = max(min(next_conductivity, 10), 0)
        
        # Occasionally introduce sudden changes (e.g., due to irrigation or environmental events)
        if np.random.rand() < 0.1:  # 10% chance of a sudden change
            next_conductivity += np.random.choice([-0.5, 0.5])  # Sudden change in conductivity
        
        # Store the simulated conductivity value
        with data_lock:
            condictive_data.append(next_conductivity)
        
        # Update the sequence with the new prediction (shift and append)
        sequence = np.roll(sequence, -1)
        sequence[-1] = next_scaled_value[0]
        
        # Vary the interval slightly for more realism (±10%)
        interval_variation = np.random.uniform(0.9, 1.1)
        time.sleep(interval * interval_variation)

def run_simulation():
    simulate_temperature_sensor(best_gbr, initial_sequenceT, scaler)

def run_humidity_simulation():
    simulate_soil_humidity_sensor(humidity_model, initial_sequenceH, scaler)

def run_condective_simulation():
    simulate_electrical_conductivity_sensor(condic_model, initial_sequenceD, scaler)

@app.route('/')
def home():
    return "Welcome to the Movie Recommendation API!"

@app.route('/start_temp_simulation', methods=['GET'])
def start_simulation():
    simulated_data.clear()

    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.start()

    return Response(generate_data(), content_type='text/event-stream')

@app.route('/start_humidity_simulation', methods=['GET'])
def start_humidity_simulation():
    humidity_data.clear()
    simulation_thread = threading.Thread(target=run_humidity_simulation)
    simulation_thread.start()
    return Response(generate_data("humidity"), content_type='text/event-stream')

@app.route('/start_conductivity_simulation', methods=['GET'])
def start_conductivity_simulation():
    humidity_data.clear()
    simulation_thread = threading.Thread(target=run_condective_simulation)
    simulation_thread.start()
    return Response(generate_data("condec"), content_type='text/event-stream')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Recommends similar movies based on the provided movie title.
    
    Expects JSON input: {"title": "movie_name"}
    """
    data = request.get_json()
    movie_title = data.get('title')

    
    if not movie_title:
        return jsonify({"error": "Movie title not provided."}), 400
    

    try:
        
        movie_index = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
        movie_index = int(movie_index)  # Convert the index to a native Python integer type
    except IndexError:
        return jsonify({"error": f"Movie '{movie_title}' not found in the dataset."}), 404
    except Exception as e:
        print(f"An unexpected error occurred while searching for the movie: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

    print(f"Movie Index: {movie_index} (Type: {type(movie_index)})")

    try:
       
        if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            return jsonify({"error": "Invalid similarity matrix shape. It must be square."}), 500

        
        similarity_scores = list(enumerate(similarity_matrix[movie_index]))
        print(f"Similarity Scores: {similarity_scores[:5]}")  # Print first 5 similarity scores for debugging


        sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]
        print(f"Sorted Similar Movies: {sorted_similar_movies}")


        recommended_titles = [movies.iloc[i[0]].title for i in sorted_similar_movies]

        return jsonify({"recommendations": recommended_titles})

    except Exception as e:
        print(f"An error occurred while processing the similarity matrix: {e}")
        return jsonify({"error": "An error occurred while processing the similarity matrix."}), 500

def generate_data(sensor_type="temperature"):
    while True:
        with data_lock:
            if sensor_type == "temperature" and simulated_data:
                data = {"temperature": simulated_data[-1]}
            elif sensor_type == "humidity" and humidity_data:
                data = {"humidity": humidity_data[-1]}
            elif sensor_type == "condec" and condictive_data:
                data={"conductivity":condictive_data[-1]}
            else:
                data = {}
            yield f"data: {json.dumps(data)}\n\n"
        time.sleep(1)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, threaded=True)
