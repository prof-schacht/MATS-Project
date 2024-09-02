from fastapi import FastAPI
import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Initialize an empty dictionary to store all features by their index
all_features = {}

# Define the folder path where JSON files are stored
folder_path = '/proj/neuronpedia_outputs/microsoft/Phi-3-mini-4k-instruct_local_blocks.16.hook_resid_post/'
pickle_file_path = '/proj/COAI-SAEDashboard/features_data.pkl'

# Check if the pickle file exists
if os.path.exists(pickle_file_path):
    print("Loading data from pickle file")
    # Load data from the pickle file
    with open(pickle_file_path, 'rb') as f:
        all_features = pickle.load(f)
else:
    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    # Load all JSON files and store their features
    for file_name in tqdm(json_files, desc="Loading JSON files"):
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'r') as file:
            data = json.load(file)
            for feature in data.get('features', []):
                feature_index = feature['feature_index']
                # Ensure all dictionaries have at least one field
                if not feature:
                    feature['dummy_field'] = None
                all_features[feature_index] = feature
    
    # Save the loaded features to a pickle file
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(all_features, f)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Feature API"}

@app.get("/features")
async def get_features():
    return {"total_features": len(all_features)}

@app.get("/feature/{feature_index}")
async def get_feature(feature_index: int):
    feature = all_features.get(feature_index)
    if feature:
        return feature
    return {"error": "Feature not found"}

@app.get("/feature_activations_with_values/{feature_index}")
async def feature_activations(feature_index: int, top_activations: bool = False, shorten_text: bool = False):
    feature_data = all_features.get(feature_index)
    if feature_data:
        activations = feature_data['activations']

        if top_activations:
            activations = sorted(activations, key=lambda x: max(x['values']), reverse=True)[:10]

        outputs = []

        for activation in activations:
            max_value = max(activation['values'])
            tokens_and_values = list(zip(activation['tokens'], activation['values']))

            if shorten_text:
                max_activation_index = activation['values'].index(max_value)
                start = max(0, max_activation_index - 15)
                end = min(len(tokens_and_values), max_activation_index + 17)
                tokens_and_values = tokens_and_values[start:end]

            output = ""
            if shorten_text and start > 0:
                output += "... "

            for token, value in tokens_and_values:
                token = ' ' + token[1:] if token.startswith('▁') else token
                output += f'{token}({value:.2f})'

            if shorten_text and end < len(activation['tokens']):
                output += " ..."

            outputs.append(output)
            outputs.append('\n\n')
            return ''.join(outputs)
    return {"error": "Feature not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
