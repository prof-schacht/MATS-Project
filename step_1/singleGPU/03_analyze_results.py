import argparse
import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from sae_lens.training.training_sae import TrainingSAE
from sae_lens import HookedSAETransformer
import pandas as pd
from jinja2 import Template
from tqdm import tqdm


def load_sae(checkpoint_path, device="cuda:0"):
    return TrainingSAE.load_from_pretrained(path=checkpoint_path, device=device)

def analyze_sparsity(sae, gpu_device):
    input_data = torch.randn(1000, sae.cfg.d_in, device=gpu_device)
    feature_acts = sae.encode_standard(input_data)
    sparsity = (feature_acts == 0).float().mean()
    return sparsity.item()

def visualize_sae_features(sae, num_features=49, grid_size=(7, 7)):
    plt.figure(figsize=(10, 10))
    feature_indices = np.random.choice(sae.W_dec.shape[0], num_features, replace=False)
    
    for i, idx in enumerate(feature_indices):
        plt.subplot(grid_size[0], grid_size[1], i+1)
        feature_vector = sae.W_dec[idx].detach().cpu().numpy()
        feature_size = feature_vector.shape[0]
        width = int(np.sqrt(feature_size))
        height = feature_size // width
        if width * height != feature_size:
            height += 1
        feature_image = np.zeros((height * width,))
        feature_image[:feature_size] = feature_vector
        feature_image = feature_image.reshape(height, width)
        plt.imshow(feature_image)
        plt.axis('off')
    plt.tight_layout()
    return plt

def analyze_feature_sparsity(sae, gpu_device):
    input_data = torch.randn(1000, sae.cfg.d_in, device=gpu_device)
    feature_acts = sae.encode_standard(input_data)
    feature_sparsity = (feature_acts == 0).float().mean(dim=0)
    overall_sparsity = feature_sparsity.mean().item()
    
    plt.figure(figsize=(10, 6))
    plt.hist(feature_sparsity.cpu().numpy(), bins=20, range=(0, 1))
    plt.title(f"Distribution of Feature Sparsities")
    plt.xlabel("Sparsity")
    plt.ylabel("Number of Features")
    return plt, overall_sparsity

def plot_feature_activation_distribution(sae, gpu_device):
    input_data = torch.randn(100, sae.cfg.d_in, device=gpu_device)
    feature_acts = sae.encode_standard(input_data)
    
    plt.figure(figsize=(10, 5))
    plt.hist(feature_acts.flatten().detach().cpu().numpy(), bins=50)
    plt.title("Distribution of Feature Activations")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    return plt

def visualize_sae_reconstruction(sae, gpu_device):
    input_data = torch.randn(1, sae.cfg.d_in, device=gpu_device)
    reconstructed = sae(input_data)

    input_reshaped = input_data.view(-1).detach().cpu().numpy()
    reconstructed_reshaped = reconstructed.view(-1).detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    im1 = ax1.imshow(input_reshaped.reshape(1, -1), aspect='auto', cmap='viridis')
    ax1.set_title("Original")
    plt.colorbar(im1, ax=ax1, label='Value')
    
    im2 = ax2.imshow(reconstructed_reshaped.reshape(1, -1), aspect='auto', cmap='viridis')
    ax2.set_title("Reconstructed")
    plt.colorbar(im2, ax=ax2, label='Value')
    
    plt.tight_layout()
    return plt

def get_top_logits(sae, model, num_features=10):
    projection_onto_unembed = sae.W_dec @ model.W_U
    vals, inds = torch.topk(projection_onto_unembed, 10, dim=1)
    random_indices = torch.randint(0, projection_onto_unembed.shape[0], (num_features,))
    top_10_logits = [model.to_str_tokens(i) for i in inds[random_indices]]
    return pd.DataFrame(top_10_logits, index=random_indices.tolist()).T

def analyze_sae(sae_path, model, gpu_device):
    sae = load_sae(sae_path, gpu_device)
    
    results = {
        "name": os.path.basename(os.path.dirname(sae_path)),
        "sparsity": analyze_sparsity(sae, gpu_device),
        "feature_viz": visualize_sae_features(sae),
        "feature_sparsity_plot, overall_sparsity": analyze_feature_sparsity(sae, gpu_device),
        "activation_dist": plot_feature_activation_distribution(sae, gpu_device),
        "reconstruction": visualize_sae_reconstruction(sae, gpu_device),
        "top_logits": None, #get_top_logits(sae, model)
        "config": sae.cfg.to_dict()  # Add SAE configuration
    }
    
    return results

def save_plot(plt, filename):
    plt.savefig(filename)
    plt.close()

def generate_html(results):
    template = Template("""
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid black; padding: 8px; text-align: left; }
            img { max-width: 300px; }
        </style>
    </head>
    <body>
        <table>
            <tr>
                <th>Metric</th>
                {% for result in results %}
                <th>{{ result['name'] }}</th>
                {% endfor %}
            </tr>
            <tr>
                <td>Sparsity</td>
                {% for result in results %}
                <td>{{ result['sparsity'] }}</td>
                {% endfor %}
            </tr>
            <tr>
                <td>Feature Visualization</td>
                {% for result in results %}
                <td><img src="{{ result['feature_viz'] }}"></td>
                {% endfor %}
            </tr>
            <tr>
                <td>Feature Sparsity Distribution</td>
                {% for result in results %}
                <td>
                    <img src="{{ result['feature_sparsity_plot'] }}">
                    <p>Overall Sparsity: {{ result['overall_sparsity'] }}</p>
                </td>
                {% endfor %}
            </tr>
            <tr>
                <td>Activation Distribution</td>
                {% for result in results %}
                <td><img src="{{ result['activation_dist'] }}"></td>
                {% endfor %}
            </tr>
            <tr>
                <td>Reconstruction</td>
                {% for result in results %}
                <td><img src="{{ result['reconstruction'] }}"></td>
                {% endfor %}
            </tr>
            <tr>
                <td>Configuration</td>
                {% for result in results %}
                <td><pre>{{ result['config'] | tojson(indent=2) }}</pre></td>
                {% endfor %}
            </tr>
        </table>
    </body>
    </html>
    """)

# ###            <!-- 
#             <tr>
#                 <td>Top Logits</td>
#                 {% for result in results %}
#                 <td>{{ result['top_logits'].to_html() | safe }}</td>
#                 {% endfor %}
#             </tr>
#             -->
    
    return template.render(results=results)

def main(checkpoint_folder, output_folder, gpu_device):
    model = None # model = HookedSAETransformer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device=gpu_device)
    
    print(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    results = []
    
    run_folders = [f for f in os.listdir(checkpoint_folder) if os.path.isdir(os.path.join(checkpoint_folder, f))]
    
    for run_folder in tqdm(run_folders, desc="Analyzing SAEs"):
        run_path = os.path.join(checkpoint_folder, run_folder)
        if os.path.isdir(run_path):
            # Find all subdirectories in the run_path
            subdirs = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
            
            # If there are subdirectories, use the first one as the sae_path
            if subdirs:
                sae_path = os.path.join(run_path, subdirs[0])
            else:
                # If no subdirectories, use the run_path itself
                sae_path = run_path
            if os.path.exists(sae_path):
                result = analyze_sae(sae_path, model, gpu_device)
                
                # Save plots as images
                feature_viz_filename = f"{run_folder}_feature_viz.png"
                feature_viz_filename_folder = os.path.join(output_folder,feature_viz_filename )
                save_plot(result['feature_viz'], feature_viz_filename_folder)
                result['feature_viz'] = feature_viz_filename
                
                feature_sparsity_plot, result['overall_sparsity'] = result['feature_sparsity_plot, overall_sparsity']
                feature_sparsity_filename = f"{run_folder}_feature_sparsity.png"
                feature_sparsity_filename_folder = os.path.join(output_folder, feature_sparsity_filename)
                save_plot(feature_sparsity_plot, feature_sparsity_filename_folder)
                result['feature_sparsity_plot'] = feature_sparsity_filename
                
                activation_dist_filename = f"{run_folder}_activation_dist.png"
                activation_dist_filename_folder = os.path.join(output_folder, activation_dist_filename)
                save_plot(result['activation_dist'], activation_dist_filename_folder)
                result['activation_dist'] = activation_dist_filename
                
                reconstruction_filename = f"{run_folder}_reconstruction.png"
                reconstruction_filename_folder = os.path.join(output_folder, reconstruction_filename)
                save_plot(result['reconstruction'], reconstruction_filename_folder)
                result['reconstruction'] = reconstruction_filename
                
                results.append(result)
    
    html_content = generate_html(results)
    
    output_file = os.path.join(output_folder, "output.html")
    
    with open(output_file, 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SAEs in a checkpoint folder")
    parser.add_argument("checkpoint_folder", nargs='?', default="./checkpoints/", help="Path to the folder containing SAE checkpoints (default: ./checkpoints/)")
    parser.add_argument("output_folder", nargs='?', default="./analyze_results/", help="Path to the output HTML file (default: ./analyze_results/)")
    parser.add_argument("gpu_device", nargs='?', default="cuda:0", help="Give name of cuda device for e.g. cuda:0 (default: cuda:0)")
    args = parser.parse_args()
    
    main(args.checkpoint_folder, args.output_folder, args.gpu_device)