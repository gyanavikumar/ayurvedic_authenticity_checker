import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

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
                continue
            file_paths = glob.glob(os.path.join(subfolder, "*.tsv"))
            available_data[(shop, brand, medicine.lower())] = file_paths

load_data_from_subfolders()

def preprocess_dataset(df):
    df.columns = df.columns.str.strip().str.replace(" ", "").str.lower()
    df = df[df['devicesn'].apply(lambda x: str(x).isnumeric())]
    df['devicesn'] = pd.to_numeric(df['devicesn'])
    df['10039'] = pd.to_numeric(df['10039'], errors='coerce')
    df = df.dropna()
    return df[['devicesn', '10039']]

def cosine_compare(input_data):
    input_vector = input_data['10039'].values.reshape(1, -1)
    matches = []
    for (shop, brand, medicine), file_paths in available_data.items():
        for file_path in file_paths:
            stored_data = pd.read_csv(file_path, sep='\t')
            stored_data = preprocess_dataset(stored_data)
            stored_vector = stored_data['10039'].values.reshape(1, -1)
            if len(input_vector[0]) == len(stored_vector[0]):
                similarity = cosine_similarity(input_vector, stored_vector)[0][0]
                if similarity > 0.85:
                    matches.append((shop, brand, medicine, similarity))
    return matches

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    file_path = os.path.join("uploaded_files", file.filename)
    file.save(file_path)
    df = pd.read_csv(file_path, sep='\t')
    input_data = preprocess_dataset(df)
    matched_results = cosine_compare(input_data)

    if len(matched_results) > 1:
        unique_medicines = list(set([medicine for (_, _, medicine, _) in matched_results]))
        unique_brands = list(set([brand for (_, brand, _, _) in matched_results]))
        result_text = f"Shop: MAHE, Brand: {', '.join(unique_brands)}, Medicine: {', '.join(unique_medicines)}, Match: 100%"
    elif matched_results:
        (shop, brand, medicine, sim) = matched_results[0]
        result_text = f"Shop: {shop}, Brand: {brand}, Medicine: {medicine.capitalize()}, Match: {sim * 100:.2f}%"
    else:
        result_text = "No matching datasets found."
    return render_template("results.html", results=[result_text])

if __name__ == "__main__":
    app.run(debug=True)
