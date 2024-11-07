import gradio as gr
import pandas as pd
import numpy as np
import torch
import pickle
from utils import (scale_data, encode_data, create_clusters, create_tsne_plot,
                  FixedNeuronsNetwork, train_model, create_data_loader, preprocess_data)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_models():
    # Load KMeans model
    with open('kmeans_model.pkl', 'rb') as f:
        clustering_model = pickle.load(f)
    
    # Load Neural Network model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = 12  # Adjust based on your feature set
    model = FixedNeuronsNetwork(num_features=num_features, 
                               num_classes=1, 
                               num_hidden_layers=3, 
                               hidden_neurons=64)
    model.load_state_dict(torch.load('neural_network_model.pth', map_location=device))
    model.eval()
    
    return clustering_model, model

def process_single_entry(customer_id, name, age, gender, location, subscription_length, 
                        monthly_bill, internet_usage, tickets, talktime, social_class, 
                        subscription_type, base_charge):
    
    # Create DataFrame from single entry
    data = pd.DataFrame({
        'CustomerID': [customer_id],
        'Name': [name],
        'Age': [age],
        'Gender': [gender],
        'Location': [location],
        'Subscription_Length_Months': [subscription_length],
        'Monthly_Bill': [monthly_bill],
        'Average_Internet_Usage': [internet_usage],
        'No_of_Tickets': [tickets],
        'Average_Talktime_Usage': [talktime],
        'Social_Class': [social_class],
        'Subscription_Type': [subscription_type],
        'Base_Charge': [base_charge]
    })
    
    # Load existing models
    clustering_model, nn_model = load_models()
    
    # Encode categorical variables
    data = encode_data(data)
    
    # Scale data for clustering
    scaled_data = scale_data(data)
    
    # Predict cluster using existing model
    cluster = clustering_model.predict(scaled_data)[0]
    
    # Prepare data for churn prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scale features for neural network
    features = data[['Age', 'Subscription_Length_Months', 'Monthly_Bill', 
                    'Average_Internet_Usage', 'No_of_Tickets', 
                    'Average_Talktime_Usage', 'Base_Charge',
                    'Subscription_Type_Encoded', 'Location_Encoded', 'Gender_Encoded']]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    input_tensor = torch.FloatTensor(scaled_features).to(device)
    
    # Get churn prediction
    with torch.no_grad():
        churn_prob = nn_model(input_tensor).cpu().numpy()[0][0]
    
    result = f"""
    Analysis Results for Customer {name}:
    Cluster Assignment: Cluster {cluster}
    Churn Prediction: {"Likely to Churn" if churn_prob > 0.5 else "Not Likely to Churn"}
    Churn Probability: {churn_prob:.2%}
    """
    
    return result

def process_csv(csv_file, train_new_models=False):
    data = pd.read_csv(csv_file.name)
    
    if train_new_models:

        data = encode_data(data)
        data = create_clusters(data)
        
        plot = create_tsne_plot(data)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        data = data[['Age', 'Subscription_Length_Months', 'Monthly_Bill',
                   'Average_Internet_Usage', 'No_of_Tickets', 'Average_Talktime_Usage',
                   'Social_Class', 'Base_Charge', 'Subscription_Type_Encoded', 'Cluster',
                   'Location_Encoded', 'Gender_Encoded']]
        
        X_train_tensor,X_train_tensor,y_train_tensor,y_test_tensor = preprocess_data(data)

        train_loader, test_loader = create_data_loader(
            X_train_tensor, X_test_tensor, 
            y_train_tensor, y_test_tensor, 
            batch_size=512
        )
        
        model = FixedNeuronsNetwork(
            num_features=X_train_scaled.shape[1],
            num_classes=1,
            num_hidden_layers=3,
            hidden_neurons=64
        ).to(device)
        
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        model, _, _, train_f1, val_f1 = train_model(
            model, criterion, optimizer, scheduler,
            train_loader, test_loader,
            num_epochs=100,
            device=device,
            patience=10
        )
        
        torch.save(model.state_dict(), 'neural_network_model.pth')
        with open('kmeans_model.pkl', 'wb') as f:
            pickle.dump(clustering_model, f)
            
    else:
        clustering_model, model = load_models()
        data = encode_data(data)
        data['Cluster'] = clustering_model.predict(scale_data(data))
        plot = create_tsne_plot(data)
    
    cluster_stats = data.groupby('Cluster')['churn_label'].mean().round(3)
    churn_distribution = data['churn_label'].value_counts(normalize=True).round(3)
    
    stats_text = f"""
    Cluster-wise Churn Probability:
    {cluster_stats.to_string()}
    
    Overall Churn Distribution:
    Non-Churners: {(1 - churn_distribution[1]):.2%}
    Churners: {churn_distribution[1]:.2%}
    """
    
    if train_new_models:
        stats_text += f"""
        
        Model Training Results:
        Training F1 Score: {train_f1:.4f}
        Validation F1 Score: {val_f1:.4f}
        """
    
    return stats_text, plot

with gr.Blocks() as app:
    gr.Markdown("# Customer Churn Analysis")
    
    with gr.Tabs():
        with gr.TabItem("Single Entry Analysis"):
            with gr.Row():
                with gr.Column():
                    customer_id = gr.Textbox(label="Customer ID")
                    name = gr.Textbox(label="Name")
                    age = gr.Number(label="Age")
                    gender = gr.Dropdown(choices=["Male", "Female"], label="Gender")
                    location = gr.Dropdown(
                        choices=["Rural", "Suburban", "Urban"], 
                        label="Location"
                    )
                    subscription_length = gr.Number(label="Subscription Length (Months)")
                    
                with gr.Column():
                    monthly_bill = gr.Number(label="Monthly Bill")
                    internet_usage = gr.Number(label="Average Internet Usage")
                    tickets = gr.Number(label="Number of Tickets")
                    talktime = gr.Number(label="Average Talktime Usage")
                    social_class = gr.Dropdown(
                        choices=["1", "2", "3"], 
                        label="Social Class",
                        value="2"
                    )
                    subscription_type = gr.Dropdown(
                        choices=["A", "B", "C", "D", "E"], 
                        label="Subscription Type",
                        value="B"
                    )
                    base_charge = gr.Number(label="Base Charge")
            
            single_submit = gr.Button("Analyze")
            single_output = gr.Textbox(label="Analysis Results")
            
        with gr.TabItem("Batch Analysis"):
            file_input = gr.File(label="Upload CSV file")
            train_new = gr.Checkbox(label="Train New Models")
            batch_submit = gr.Button("Analyze Batch")
            with gr.Row():
                batch_output = gr.Textbox(label="Batch Analysis Results")
                plot_output = gr.Plot(label="Cluster Visualization")
    
    # Set up event handlers
    single_submit.click(
        fn=process_single_entry,
        inputs=[customer_id, name, age, gender, location, subscription_length,
                monthly_bill, internet_usage, tickets, talktime, social_class,
                subscription_type, base_charge],
        outputs=single_output
    )
    
    batch_submit.click(
        fn=process_csv,
        inputs=[file_input, train_new],
        outputs=[batch_output, plot_output]
    )

if __name__ == "__main__":
    app.launch()