import pandas as pd
import matplotlib.pyplot as plt
import os

# Create outputs folder if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')
    print("Created 'outputs' folder.")

try:
    # Load datasets
    print("Loading datasets...")
    fake_df = pd.read_csv('dataset/Fake.csv')
    true_df = pd.read_csv('dataset/True.csv')
    print("✓ Datasets loaded successfully.\n")
    
    # Add label column to distinguish fake vs real
    fake_df['label'] = 'Fake'
    true_df['label'] = 'Real'
    
    # Combine datasets
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # ===== DISPLAY DATASET INFORMATION =====
    print("=" * 60)
    print("DATASET INFORMATION - FAKE NEWS")
    print("=" * 60)
    print(f"Shape: {fake_df.shape}")
    print(f"Columns: {fake_df.columns.tolist()}")
    print("\nFirst 5 Rows:")
    print(fake_df.head())
    print("\nDataset Info:")
    print(fake_df.info())
    print(f"\nMissing Values:\n{fake_df.isnull().sum()}")
    
    print("\n" + "=" * 60)
    print("DATASET INFORMATION - TRUE NEWS")
    print("=" * 60)
    print(f"Shape: {true_df.shape}")
    print(f"Columns: {true_df.columns.tolist()}")
    print("\nFirst 5 Rows:")
    print(true_df.head())
    print("\nDataset Info:")
    print(true_df.info())
    print(f"\nMissing Values:\n{true_df.isnull().sum()}")
    
    # ===== CHECK FOR DUPLICATES =====
    print("\n" + "=" * 60)
    print("DUPLICATE RECORDS CHECK")
    print("=" * 60)
    fake_duplicates = fake_df.duplicated().sum()
    true_duplicates = true_df.duplicated().sum()
    combined_duplicates = combined_df.duplicated().sum()
    
    print(f"Duplicates in Fake News: {fake_duplicates}")
    print(f"Duplicates in True News: {true_duplicates}")
    print(f"Duplicates in Combined Dataset: {combined_duplicates}")
    
    # ===== COMPARE FAKE VS REAL NEWS COUNT =====
    print("\n" + "=" * 60)
    print("FAKE VS REAL NEWS COUNT")
    print("=" * 60)
    fake_count = len(fake_df)
    true_count = len(true_df)
    total_count = fake_count + true_count
    
    print(f"Fake News Count: {fake_count}")
    print(f"Real News Count: {true_count}")
    print(f"Total News Count: {total_count}")
    print(f"Fake News Percentage: {(fake_count / total_count) * 100:.2f}%")
    print(f"Real News Percentage: {(true_count / total_count) * 100:.2f}%")
    
    # ===== GENERATE VISUALIZATIONS =====
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Bar Chart
    plt.figure(figsize=(10, 6))
    categories = ['Fake News', 'Real News']
    counts = [fake_count, true_count]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = plt.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('News Type', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Fake vs Real News Count - Bar Chart', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/fake_vs_real_bar_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Bar chart saved to 'outputs/fake_vs_real_bar_chart.png'")
    plt.close()
    
    # Pie Chart
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=categories, autopct='%1.2f%%', colors=colors,
            startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
            explode=(0.05, 0.05), shadow=True)
    plt.title('Fake vs Real News Distribution - Pie Chart', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/fake_vs_real_pie_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Pie chart saved to 'outputs/fake_vs_real_pie_chart.png'")
    plt.close()
    
    # ===== SUMMARY REPORT =====
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    print(f"✓ Fake News Dataset: {fake_count} records, {fake_df.shape[1]} columns")
    print(f"✓ Real News Dataset: {true_count} records, {true_df.shape[1]} columns")
    print(f"✓ Total Records: {total_count}")
    print(f"✓ Duplicate Records: {combined_duplicates}")
    print(f"✓ Missing Values: {combined_df.isnull().sum().sum()}")
    print(f"✓ Visualizations: 2 charts generated and saved to 'outputs/'")
    print("=" * 60)
    print("Analysis completed successfully!")
    print("=" * 60)
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure 'dataset/Fake.csv' and 'dataset/True.csv' exist.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
