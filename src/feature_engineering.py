"""
Feature Engineering Module
Converts raw text data into TF-IDF features and splits dataset into training and testing sets.
"""

import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_and_prepare_data():
    """
    Load processed news data and prepare features.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, tfidf_vectorizer)
    """
    try:
        # Load the processed dataset
        print("=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        data_path = "processed_news.csv"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
        print("\n[1] Loading dataset...")
        df = pd.read_csv(data_path)
        original_shape = df.shape
        print(f"✓ Dataset loaded successfully!")
        print(f"  Original dataset shape: {original_shape}")
        
        # Validate required columns
        if 'content' not in df.columns:
            raise ValueError("'content' column not found in dataset!")
        
        if 'label' not in df.columns:
            raise ValueError("'label' column not found in dataset!")
        
        # Check for missing values
        if df['content'].isnull().any():
            print(f"  Warning: Found {df['content'].isnull().sum()} null values in 'content' column")
            df = df.dropna(subset=['content'])
            print(f"  After removing nulls: {df.shape}")
        
        # Separate features and labels
        X = df['content']
        y = df['label']
        
        print("\n[2] Converting text to TF-IDF features...")
        # Initialize TF-IDF vectorizer with max_features = 5000
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        
        # Fit and transform the text data
        X_tfidf = tfidf_vectorizer.fit_transform(X)
        print(f"✓ TF-IDF features created successfully!")
        print(f"  TF-IDF feature matrix shape: {X_tfidf.shape}")
        
        print("\n[3] Splitting dataset (80% training, 20% testing)...")
        # Split dataset into training and testing sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        print(f"✓ Dataset split successfully!")
        print(f"  Training set shape: {X_train.shape}")
        print(f"  Testing set shape: {X_test.shape}")
        
        print("\n[4] Saving artifacts to models folder...")
        # Ensure models directory exists
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"  Created models directory: {models_dir}")
        
        # Save training features
        joblib.dump(X_train, os.path.join(models_dir, "X_train.pkl"))
        print("  ✓ X_train.pkl saved")
        
        # Save testing features
        joblib.dump(X_test, os.path.join(models_dir, "X_test.pkl"))
        print("  ✓ X_test.pkl saved")
        
        # Save training labels
        joblib.dump(y_train, os.path.join(models_dir, "y_train.pkl"))
        print("  ✓ y_train.pkl saved")
        
        # Save testing labels
        joblib.dump(y_test, os.path.join(models_dir, "y_test.pkl"))
        print("  ✓ y_test.pkl saved")
        
        # Save the TF-IDF vectorizer for future use
        joblib.dump(tfidf_vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        print("  ✓ tfidf_vectorizer.pkl saved")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Original dataset shape:        {original_shape}")
        print(f"TF-IDF feature matrix shape:   {X_tfidf.shape}")
        print(f"Training set shape:            {X_train.shape}")
        print(f"Testing set shape:             {X_test.shape}")
        print(f"Training samples:              {X_train.shape[0]}")
        print(f"Testing samples:               {X_test.shape[0]}")
        print(f"Number of TF-IDF features:     {X_tfidf.shape[1]}")
        print("=" * 60)
        print("\n✓ Feature engineering pipeline completed successfully!")
        
        return X_train, X_test, y_train, y_test, tfidf_vectorizer
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure 'processed_news.csv' exists in the project root directory.")
        raise
    
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("Please check the dataset structure and column names.")
        raise
    
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    try:
        # Run the feature engineering pipeline
        X_train, X_test, y_train, y_test, vectorizer = load_and_prepare_data()
    
    except Exception as e:
        print(f"\nFeature engineering failed. Error: {e}")
        exit(1)
