import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. SETUP & LOADING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Customer Prediction", layout="wide")

@st.cache_resource
def load_resources():
    # Load the trained model and encoders
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        return None, None

def main():
    st.title("🛍️ Customer Subscription Predictor")
    
    # Load raw data for display (optional, just for visuals)
    try:
        raw_df = pd.read_csv('C:\\Users\\DELL\\Downloads\\shopping_behavior_updated.csv')
    except:
        st.error("CSV file not found.")
        return

    model, encoders = load_resources()

    if not model:
        st.error("Model not found! Please run 'model.py' first to generate the model files.")
        st.stop()

    # -------------------------------------------------------------------------
    # 2. DASHBOARD (Brief)
    # -------------------------------------------------------------------------
    st.subheader("📊 Data Insights")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(raw_df, x="Age", title="Age Distribution", color_discrete_sequence=['#3366CC'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.pie(raw_df, names="Subscription Status", title="Subscription Status Split")
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------------
    # 3. PREDICTION INTERFACE
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("🤖 Predict New Customer")

    # We need input fields for every feature the model was trained on
    # We use the encoders to know what options were available during training
    
    # Get feature names expected by the model
    feature_names = model.feature_names_in_
    
    input_data = {}
    
    with st.form("prediction_form"):
        cols = st.columns(3)
        for i, col in enumerate(feature_names):
            # If the column has an encoder, it's categorical -> Use Selectbox
            if col in encoders:
                # We don't need to predict the target, but it won't be in feature_names anyway
                options = encoders[col].classes_
                val = cols[i % 3].selectbox(f"Select {col}", options)
                # Transform inputs immediately using the encoder
                input_data[col] = encoders[col].transform([val])[0]
            else:
                # Numeric column -> Use Number Input
                # Default value is the mean of that column in raw data
                default_val = float(raw_df[col].mean())
                val = cols[i % 3].number_input(f"Enter {col}", value=default_val)
                input_data[col] = val
        
        submitted = st.form_submit_button("Predict Subscription Status")

    if submitted:
        # Create DataFrame for model
        input_df = pd.DataFrame([input_data])
        
        # Predict
        prediction_code = model.predict(input_df)[0]
        
        # Decode prediction back to text (Yes/No)
        prediction_label = encoders['Subscription Status'].inverse_transform([prediction_code])[0]
        
        if prediction_label == 'Yes':
            st.success("## ✅ Prediction: Customer will likely SUBSCRIBE")
        else:
            st.warning("## ❌ Prediction: Customer will likely NOT SUBSCRIBE")

if __name__ == "__main__":
    main()