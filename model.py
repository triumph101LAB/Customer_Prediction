import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_model():
    # 1. Load Data
    try:
        df = pd.read_csv('C:\\Users\\DELL\\Downloads\\shopping_behavior_updated.csv')
    except FileNotFoundError:
        print("Error: 'shopping_behavior_updated.csv' not found.")
        return

    # 2. Preprocessing
    # Drop ID as it is not useful for prediction
    df = df.drop(columns=['Customer ID'])
    
    # Create a dictionary to store encoders for each column
    encoders = {}
    
    # Encode all categorical columns (text -> numbers)
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # 3. Define Features (X) and Target (y)
    target_col = 'Subscription Status'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 4. Train Model
    print("Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 5. Save Model and Encoders
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    print("Success! 'model.pkl' and 'encoders.pkl' have been saved.")

if __name__ == "__main__":
    train_model()