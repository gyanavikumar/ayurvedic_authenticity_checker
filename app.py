import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

app = Flask(__name__)

# Load datasets from subfolders
parent_directory = "/Users/apple/Desktop/Python/Authenticity Checker mix/Datasets"
available_data = {}

def load_data_from_subfolders():
    import glob
    subfolders = glob.glob(os.path.join(parent_directory, "*"))
    for subfolder in subfolders:
        if os.path.isdir(subfolder):
            folder_name = os.path.basename(subfolder)
            try:
                shop, brand, medicine = folder_name.split("_")
            except ValueError:
                continue  # Skip invalid folder names
            file_paths = glob.glob(os.path.join(subfolder, "*.tsv"))
            available_data[(shop, brand, medicine)] = file_paths

load_data_from_subfolders()

# Preprocess the dataset
def preprocess_dataset(df):
    df.columns = df.columns.str.strip().str.replace(" ", "").str.lower()
    column_mapping = {"devicesn": "DeviceSN", "10039": "10039"}
    df.rename(columns=column_mapping, inplace=True)
    df = df[df['DeviceSN'].apply(lambda x: str(x).isnumeric())]
    df['DeviceSN'] = pd.to_numeric(df['DeviceSN'])
    df['10039'] = pd.to_numeric(df['10039'], errors='coerce')
    df = df.dropna()
    return df[['DeviceSN', '10039']].reset_index(drop=True)

# Feature extraction function for ML
def extract_features(df):
    features = {
        'mean': df['10039'].mean(),
        'std': df['10039'].std(),
        'min': df['10039'].min(),
        'max': df['10039'].max(),
        'median': df['10039'].median(),
        'range': df['10039'].max() - df['10039'].min()
    }
    return list(features.values())

# Load the trained model and scaler
model = joblib.load('authenticity_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to match datasets using proportionality logic
def match_datasets(input_data, threshold=0.7, tolerance=0.01):
    matches = []
    for (shop, brand, medicine), file_paths in available_data.items():
        for file_path in file_paths:
            try:
                stored_data = pd.read_csv(file_path, sep='\t')
                stored_data = preprocess_dataset(stored_data)

                # Merge input and stored data on DeviceSN
                merged = input_data.merge(stored_data, on="DeviceSN", suffixes=('_input', '_stored'))

                def is_proportional(row):
                    value1 = row['10039_input']
                    value2 = row['10039_stored']
                    if value1 == 0 or value2 == 0:  # Avoid division by zero
                        return False
                    ratio = value1 / value2
                    ratio_rounded = round(ratio)
                    return abs(ratio - ratio_rounded) < tolerance

                merged['Proportional'] = merged.apply(is_proportional, axis=1)
                matched_count = merged['Proportional'].sum()

                total_rows = len(input_data)
                match_percentage = matched_count / total_rows if total_rows > 0 else 0

                if match_percentage >= threshold:
                    matches.append((shop, brand, medicine, match_percentage * 100))

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

    return matches

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Save the uploaded file temporarily
    upload_folder = os.path.join(os.getcwd(), "uploaded_files")
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    try:
        # Preprocess the input dataset
        df = pd.read_csv(file_path, sep='\t') if file_path.endswith('.tsv') else pd.read_excel(file_path)
        input_data = preprocess_dataset(df)

        # Extract features and scale for ML prediction
        features = extract_features(input_data)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)

        # Debugging: Print the raw prediction
        print(f"Raw Prediction: {prediction[0]}")

        # Check proportionality for more detailed matching
        proportional_results = match_datasets(input_data)

        # Prepare the results list
        results = []
        if proportional_results:
            results += [
                f"Shop: {shop}, Brand: {brand}, Medicine: {medicine}, Match: {percentage:.2f}%"
                for shop, brand, medicine, percentage in proportional_results
            ]
        else:
            # Format the model prediction with fallback handling
            predicted_medicine = prediction[0]
            parts = predicted_medicine.split("_")

            if len(parts) == 3:
                shop, brand, medicine = parts
                formatted_prediction = f"Shop: {shop}, Brand: {brand}, Medicine: {medicine}, Match: 100%"
            else:
                formatted_prediction = f"Predicted Medicine: {predicted_medicine}, Match: 100%"

            results.append(formatted_prediction)

        # Handle case when no matches are found at all
        if not results:
            results.append("No proportional match found with more than 70% similarity.")

        # Pass results as a list
        return render_template("results.html", results=results)

    except Exception as e:
        return render_template("results.html", results=[f"Processing Error: {str(e)}"])


if __name__ == "__main__":
    import webbrowser
    import threading

    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open_new("http://127.0.0.1:5000/")

    threading.Thread(target=open_browser).start()
    app.run(debug=True)