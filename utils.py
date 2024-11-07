import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def scale_data(df):
    
    scaler = StandardScaler()
    features = df[['Average_Internet_Usage', 'Average_Talktime_Usage', 'Monthly_Bill', 
               'Subscription_Type_Encoded']]
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    
    return scaled_df

def encode_data(df):
    
    label_encoder = LabelEncoder()
    
    df['Subscription_Type_Encoded'] = label_encoder.fit_transform(df['Subscription_Type'])
    df['Location_Encoded'] = label_encoder.fit_transform(df['Location'])
    df["Gender_Encoded"] = label_encoder.fit_transform(df['Gender'])
    
    return df

def preprocess_data(df):

    X = df.drop(columns=['churn_label', 'CustomerID', 'Name', 'Subscription_Type', 'Gender', 'Location'])
    y = df['churn_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_train_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    return X_train_tensor,X_train_tensor,y_train_tensor,y_test_tensor
    
def create_tsne_plot(df):
    
    scaled_df = scale_data(df)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(scaled_df)

    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df_tsne['Cluster'] = df['Cluster']

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='Cluster',
        palette=sns.color_palette('hsv', len(df['Cluster'].unique())),
        data=df_tsne,
        legend='full'
    )
    plt.title('t-SNE Visualization of K Means Clustering')
    plt.show()
    
def create_clusters(df):
    
    scaled_df = scale_data(df)

    kmeans = KMeans(n_clusters=8, random_state=42)
    kmeans.fit(scaled_df)

    df['Cluster'] = kmeans.labels_
    
    return df

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

def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs, device,patience):
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    patience = patience  # Number of epochs to wait before stopping
    patience_counter = 0
    best_model = None

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        all_train_targets = []
        all_train_preds = []

        # Training loop
        with tqdm(total=len(train_loader), desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='batch') as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * inputs.size(0)

                # Store targets and predictions for F1 score calculation
                all_train_targets.extend(targets.cpu().numpy())
                all_train_preds.extend(outputs.cpu().detach().numpy())

                # Update the progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        # Calculate average training loss and F1 score for training
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)

        train_f1 = f1_score(np.squeeze(all_train_targets), np.where(np.squeeze(all_train_preds) > 0.5, 1, 0), pos_label=1, average='binary')

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        all_val_targets = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                total_val_loss += loss.item() * inputs.size(0)

                all_val_targets.extend(targets.cpu().numpy())
                all_val_preds.extend(outputs.cpu().detach().numpy())

        # Calculate average validation loss and F1 score
        avg_val_loss = total_val_loss / len(test_loader.dataset)
        val_loss_history.append(avg_val_loss)

        val_f1 = f1_score(np.squeeze(all_val_targets), np.where(np.squeeze(all_val_preds) > 0.5, 1, 0), pos_label=1, average='binary')

        # Print training and validation results
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f} | Avg Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Step the learning rate scheduler
        scheduler.step(avg_train_loss)

        # Early stopping based on F1 score and patience counter
        if val_f1 > 0.8:
            print(f"Stopping training early at epoch {epoch + 1} as validation F1 score exceeded 0.8.")
            break

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()  # Save best model weights
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    model.load_state_dict(best_model)  # Load the best model before returning
    return model, train_loss_history, val_loss_history, train_f1, val_f1

def create_data_loader(x_train, x_test, y_train, y_test, batch_size, sampling = True):
    """
    Takes input as torch tensors and creates data loaders.
    """
    
    # Create TensorDatasets and move them to the specified device
    train_dataset = TensorDataset(x_train, y_train.reshape(-1,1))
    test_dataset = TensorDataset(x_test, y_test.reshape(-1,1))
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if sampling:
        class_counts = torch.bincount(y_train.long().view(-1))
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        sample_weights = class_weights[y_train.long().view(-1)]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, shuffle=False)
    else:
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader