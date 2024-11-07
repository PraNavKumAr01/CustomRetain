import gradio as gr
import pandas as pd
from src.utilfuncs import (
    load_kmeans_model,
    load_churn_model,
    encode_and_scale,
    get_cluster,
    get_churn_label,
    load_encoder,
    load_scaler
)

def app():

    kmeans_model = load_kmeans_model('kmeans_model_main.pkl')
    churn_model = load_churn_model('neural_network_model_final.pth')
    encoders = load_encoder('label_encoders.pkl')
    scaler = load_scaler('scaler.pkl')

    with gr.Group():
        gr.Markdown("## Churn Prediction App")
        age = gr.Slider(label="Age", minimum=18, maximum=100, step=1, value=30)
        gender = gr.Dropdown(label="Gender", choices=["Male", "Female"], value="Male")
        location = gr.Dropdown(label="Location", choices=["Urban", "Rural", "Suburban"], value="Urban")
        subscription_length = gr.Slider(label="Subscription Length (Months)", minimum=1, maximum=60, step=1, value=12)
        monthly_bill = gr.Slider(label="Monthly Bill", minimum=10, maximum=1000, step=1, value=50)
        avg_internet_usage = gr.Slider(label="Average Internet Usage", minimum=1, maximum=200, step=1, value=50)
        num_tickets = gr.Slider(label="Number of Tickets", minimum=0, maximum=20, step=1, value=2)
        avg_talktime = gr.Slider(label="Average Talktime Usage", minimum=1, maximum=200, step=1, value=50)
        social_class = gr.Slider(label="Social Class", minimum = 1, maximum = 3, step = 1, value=1)
        subscription_type = gr.Dropdown(label="Subscription Type", choices=["A", "B", "C", "D", "E"], value="B")
        base_charge = gr.Slider(label="Base Charge", minimum=10, maximum=500, step=1, value=30)

    with gr.Group():
        gr.Markdown("## Results")
        predicted_group = gr.Text(label="Predicted Group")
        predicted_churn = gr.Text(label="Predicted Churn")

    def predict(age, gender, location, subscription_length, monthly_bill, avg_internet_usage, num_tickets, avg_talktime, social_class, subscription_type, base_charge):

        data = {
            'Age': [age],
            'Gender': [gender],
            'Location': [location],
            'Subscription_Length_Months': [subscription_length],
            'Monthly_Bill': [monthly_bill],
            'Average_Internet_Usage': [avg_internet_usage],
            'No_of_Tickets': [num_tickets],
            'Average_Talktime_Usage': [avg_talktime],
            'Social_Class': [social_class],
            'Subscription_Type': [subscription_type],
            'Base_Charge': [base_charge]
        }
        df = pd.DataFrame(data)
        scaled_df = encode_and_scale(df, encoders, scaler)

        group = get_cluster(scaled_df, kmeans_model)
        predicted_group_text = group[0]

        churn_label = get_churn_label(scaled_df, churn_model)
        predicted_churn_text = 'Yes' if churn_label[0][0] == 1 else 'No'

        return predicted_group_text, predicted_churn_text

    app = gr.Interface(
        fn=predict,
        inputs=[
            age, gender, location, subscription_length, monthly_bill, avg_internet_usage, num_tickets, avg_talktime, social_class, subscription_type, base_charge
        ],
        outputs=[predicted_group, predicted_churn],
        title="Churn Prediction App",
        description="Enter customer information and get predicted churn and customer group."
    )

    app.launch()

if __name__ == "__main__":
    app()