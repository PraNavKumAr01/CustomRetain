import torch
import torch.nn as nn
import joblib
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from PIL import Image

class FixedNeuronsNetwork(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden_layers, hidden_neurons, layer_activation=nn.ReLU, final_activation=nn.Sigmoid):
        super(FixedNeuronsNetwork, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(num_features, hidden_neurons))
        layers.append(layer_activation())
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(layer_activation())
        
        layers.append(nn.Linear(hidden_neurons, num_classes))
        
        self.final_activation = final_activation()
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.network(x)
        output = self.final_activation(x)
        return output
    
def load_kmeans_model(model_path):
    kmeans_loaded = joblib.load('kmeans_model_main.pkl')
    return kmeans_loaded

def load_churn_model(model_path):
    model = FixedNeuronsNetwork(num_features=11,num_classes=1,num_hidden_layers=3,hidden_neurons=64).to('cpu')
    model.load_state_dict(torch.load(model_path))

    return model

def load_encoder(model_path):
    label_encoders = joblib.load(model_path)
    return label_encoders

def load_scaler(model_path):
    scaler = joblib.load(model_path)
    return scaler

def encode_and_scale(new_sample_df, encoders, scaler):

    numerical_cols = ['Age', 'Subscription_Length_Months', 'Monthly_Bill','Average_Internet_Usage', 'No_of_Tickets', 'Average_Talktime_Usage','Social_Class', 'Base_Charge']
    new_sample_df[numerical_cols] = scaler.transform(new_sample_df[numerical_cols])

    for col, le in encoders.items():
        new_sample_df[col] = le.transform(new_sample_df[col])

    return new_sample_df

def get_cluster(sample, kmeans_loaded):
    group = kmeans_loaded.predict(sample)

    return group

def get_churn_label(sample, model):
    sample_tensor = torch.tensor(sample.values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(sample_tensor)
        predictions = (predictions > 0.5).float()

    return predictions

def create_tsne_plot(df, clusters):

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df)

    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df_tsne['Cluster'] = clusters

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='Cluster',
        palette=sns.color_palette('hsv', len(set(clusters))),
        data=df_tsne,
        legend='full'
    )
    plt.title('t-SNE Visualization of K Means Clustering')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_image = np.asarray(Image.open(buf))

    return plot_image