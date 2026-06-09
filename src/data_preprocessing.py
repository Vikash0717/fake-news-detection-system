"""
Data Preprocessing Module for Fake News Detection

This module handles loading, cleaning, and preprocessing of fake and real news datasets.
It performs text cleaning, removes stopwords, and generates processed datasets for model training.

Author: Data Science Team
Date: 2024
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import NLTK stopwords
try:
    from nltk.corpus import stopwords
    import nltk
    # Download stopwords if not already present
    nltk.download('stopwords', quiet=True)
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except ImportError as e:
    logger.warning(f"NLTK not found: {e}. Please install with: pip install nltk")
    ENGLISH_STOPWORDS = set()


class DataPreprocessor:
    """
    A class to handle data preprocessing for fake news detection.
    
    Attributes:
        fake_data_path (str): Path to Fake.csv
        true_data_path (str): Path to True.csv
        output_path (str): Path to save processed data
    """
    
    def __init__(self, fake_path: str, true_path: str, output_path: str = "processed_news.csv"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            fake_path (str): Path to Fake.csv
            true_path (str): Path to True.csv
            output_path (str): Output path for processed data (default: processed_news.csv)
        """
        self.fake_data_path = fake_path
        self.true_data_path = true_path
        self.output_path = output_path
        self.df = None
        self.original_size = 0
        self.final_size = 0
        
    def load_datasets(self) -> pd.DataFrame:
        """
        Load and combine fake and real news datasets.
        
        Returns:
            pd.DataFrame: Combined dataset with labels
            
        Raises:
            FileNotFoundError: If dataset files are not found
            Exception: If datasets cannot be loaded
        """
        try:
            logger.info("Loading datasets...")
            
            # Check if files exist
            if not os.path.exists(self.fake_data_path):
                raise FileNotFoundError(f"Fake dataset not found: {self.fake_data_path}")
            if not os.path.exists(self.true_data_path):
                raise FileNotFoundError(f"True dataset not found: {self.true_data_path}")
            
            # Load datasets
            fake_df = pd.read_csv(self.fake_data_path)
            true_df = pd.read_csv(self.true_data_path)
            
            # Add labels: 0 for fake, 1 for real
            fake_df['label'] = 0
            true_df['label'] = 1
            
            # Combine datasets
            df = pd.concat([fake_df, true_df], ignore_index=True)
            
            self.original_size = len(df)
            logger.info(f"Datasets loaded successfully. Total records: {self.original_size}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and keep only required columns: title, text, label.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with selected columns
        """
        try:
            # Check if required columns exist
            required_cols = ['title', 'text', 'label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Select only required columns
            df = df[required_cols].copy()
            logger.info(f"Selected columns: {required_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error selecting columns: {e}")
            raise
    
    def remove_missing_and_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing values and duplicate records.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        try:
            initial_size = len(df)
            
            # Remove missing values
            df = df.dropna()
            after_na_drop = len(df)
            na_removed = initial_size - after_na_drop
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['title', 'text'], keep='first')
            after_dup_drop = len(df)
            dup_removed = after_na_drop - after_dup_drop
            
            logger.info(f"Removed {na_removed} rows with missing values")
            logger.info(f"Removed {dup_removed} duplicate records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing missing values and duplicates: {e}")
            raise
    
    def create_content_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a new column by concatenating title and text.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new 'content' column
        """
        try:
            # Create content by combining title and text
            df['content'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
            logger.info("Created 'content' column from title and text")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating content column: {e}")
            raise
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by converting to lowercase, removing punctuation,
        special characters, numbers, and extra spaces.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        try:
            if not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {e}")
            return ""
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove English stopwords from text.
        
        Args:
            text (str): Cleaned text
            
        Returns:
            str: Text without stopwords
        """
        try:
            if not text:
                return ""
            
            words = text.split()
            filtered_words = [word for word in words if word not in ENGLISH_STOPWORDS]
            
            return ' '.join(filtered_words)
            
        except Exception as e:
            logger.warning(f"Error removing stopwords: {e}")
            return text
    
    def clean_and_process_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply text cleaning and stopword removal to content column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with cleaned content
        """
        try:
            logger.info("Cleaning text data...")
            
            # Apply text cleaning
            df['content'] = df['content'].apply(self.clean_text)
            
            # Remove stopwords
            df['content'] = df['content'].apply(self.remove_stopwords)
            
            # Remove empty rows created during cleaning
            df = df[df['content'].str.len() > 0]
            
            logger.info("Text cleaning completed")
            
            return df
            
        except Exception as e:
            logger.error(f"Error during text processing: {e}")
            raise
    
    def process_data(self) -> Tuple[pd.DataFrame, dict]:
        """
        Execute the complete data preprocessing pipeline.
        
        Returns:
            Tuple[pd.DataFrame, dict]: Processed dataframe and statistics
        """
        try:
            # Step 1: Load datasets
            self.df = self.load_datasets()
            
            # Step 2: Select columns
            self.df = self.select_columns(self.df)
            
            # Step 3: Remove missing values and duplicates
            self.df = self.remove_missing_and_duplicates(self.df)
            
            # Step 4: Create content column
            self.df = self.create_content_column(self.df)
            
            # Step 5: Clean and process text
            self.df = self.clean_and_process_text(self.df)
            
            # Update final size
            self.final_size = len(self.df)
            
            logger.info("Data preprocessing completed successfully")
            
            # Calculate statistics
            stats = {
                'original_size': self.original_size,
                'final_size': self.final_size,
                'rows_removed': self.original_size - self.final_size,
                'removal_percentage': round((self.original_size - self.final_size) / self.original_size * 100, 2)
            }
            
            return self.df, stats
            
        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            raise
    
    def save_processed_data(self) -> None:
        """
        Save the processed dataset to CSV file.
        
        Raises:
            ValueError: If no data to save
            Exception: If save operation fails
        """
        try:
            if self.df is None or len(self.df) == 0:
                raise ValueError("No data to save. Please run process_data() first.")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save to CSV
            self.df.to_csv(self.output_path, index=False)
            logger.info(f"Processed data saved to {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def print_statistics(self) -> None:
        """
        Print preprocessing statistics and sample data.
        """
        try:
            if self.df is None:
                logger.warning("No data to display. Please run process_data() first.")
                return
            
            print("\n" + "="*60)
            print("DATA PREPROCESSING SUMMARY")
            print("="*60)
            print(f"Original Dataset Size: {self.original_size:,}")
            print(f"Final Dataset Size: {self.final_size:,}")
            print(f"Rows Removed: {self.original_size - self.final_size:,}")
            print(f"Removal Percentage: {round((self.original_size - self.final_size) / self.original_size * 100, 2)}%")
            print("-"*60)
            print(f"Label Distribution:")
            print(self.df['label'].value_counts().to_string())
            print("="*60)
            print("\nSAMPLE CLEANED TEXT (First 3 records):")
            print("-"*60)
            
            for idx, row in self.df.head(3).iterrows():
                print(f"\nRecord {idx + 1}:")
                print(f"Label: {'Fake' if row['label'] == 0 else 'Real'}")
                print(f"Content: {row['content'][:200]}...")
            
            print("\n" + "="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Error printing statistics: {e}")


def main():
    """
    Main function to execute the data preprocessing pipeline.
    """
    try:
        # Define file paths (adjust paths as needed for your environment)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        fake_path = os.path.join(parent_dir, 'dataset', 'Fake.csv')
        true_path = os.path.join(parent_dir, 'dataset', 'True.csv')
        output_path = os.path.join(parent_dir, 'processed_news.csv')
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(fake_path, true_path, output_path)
        
        # Process data
        logger.info("Starting data preprocessing pipeline...")
        processed_df, stats = preprocessor.process_data()
        
        # Save processed data
        preprocessor.save_processed_data()
        
        # Print statistics
        preprocessor.print_statistics()
        
        logger.info("Pipeline completed successfully!")
        
        return processed_df, stats
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
