import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd


def load_dataset(csv_path):
    """Load the processed news CSV from the project root."""
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{csv_path}'.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Dataset file at '{csv_path}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError as exc:
        print(f"Error parsing CSV file '{csv_path}': {exc}")
        sys.exit(1)


def ensure_output_directory(directory):
    """Create the outputs directory if it does not exist."""
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        print(f"Error creating output directory '{directory}': {exc}")
        sys.exit(1)


def plot_class_distribution(label_counts, output_dir):
    """Generate bar and pie charts for fake vs real news distribution."""
    labels = ['Fake', 'Real']
    counts = [label_counts.get(0, 0), label_counts.get(1, 0)]

    # Bar chart
    plt.figure(figsize=(6, 5))
    plt.bar(labels, counts, color=['#d9534f', '#5cb85c'])
    plt.title('Fake vs Real News Count')
    plt.ylabel('Count')
    plt.tight_layout()
    bar_path = os.path.join(output_dir, 'class_distribution_bar.png')
    plt.savefig(bar_path)
    plt.close()

    # Pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#d9534f', '#5cb85c'], startangle=140)
    plt.title('Fake vs Real News Distribution')
    plt.tight_layout()
    pie_path = os.path.join(output_dir, 'class_distribution_pie.png')
    plt.savefig(pie_path)
    plt.close()

    return bar_path, pie_path


def get_top_words(df, label_value, text_column='content', top_n=20):
    """Return the top N most frequent words for a given label."""
    subset = df[df['label'] == label_value]
    if subset.empty:
        return []

    all_words = []
    for text in subset[text_column].dropna().astype(str):
        all_words.extend(text.split())

    word_counts = Counter(all_words)
    return word_counts.most_common(top_n)


def plot_top_words(top_words, title, output_path):
    """Plot the top words bar chart and save it."""
    if not top_words:
        print(f"Warning: No top words to plot for '{title}'.")
        return

    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color='#337ab7')
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_insights(fake_words, real_words):
    """Print insights about the most common words in fake and real news."""
    print('\nMost common fake news words:')
    for word, count in fake_words:
        print(f"  {word}: {count}")

    print('\nMost common real news words:')
    for word, count in real_words:
        print(f"  {word}: {count}")


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(project_root, 'processed_news.csv')
    output_dir = os.path.join(project_root, 'outputs')

    ensure_output_directory(output_dir)

    df = load_dataset(csv_path)

    print('Dataset Shape:')
    print(df.shape)
    print('\nColumn Names:')
    print(df.columns.tolist())
    print('\nFirst 5 Rows:')
    print(df.head(5))

    if 'label' not in df.columns:
        print("Error: The dataset does not contain a 'label' column.")
        sys.exit(1)

    label_distribution = df['label'].value_counts().sort_index()
    print('\nLabel Distribution:')
    print(label_distribution)

    plot_class_distribution(label_distribution.to_dict(), output_dir)

    fake_top_words = get_top_words(df, label_value=0, text_column='content', top_n=20)
    real_top_words = get_top_words(df, label_value=1, text_column='content', top_n=20)

    fake_output = os.path.join(output_dir, 'top_20_fake_words.png')
    real_output = os.path.join(output_dir, 'top_20_real_words.png')

    plot_top_words(fake_top_words, 'Top 20 Fake News Words', fake_output)
    plot_top_words(real_top_words, 'Top 20 Real News Words', real_output)

    print_insights(fake_top_words, real_top_words)
    print('\nSaved charts to outputs folder:')
    print(f"  {fake_output}")
    print(f"  {real_output}")
    print(f"  {os.path.join(output_dir, 'class_distribution_bar.png')}")
    print(f"  {os.path.join(output_dir, 'class_distribution_pie.png')}")


if __name__ == '__main__':
    main()
