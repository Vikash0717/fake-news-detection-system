"""
Logistic Regression Model Training Script

This script loads preprocessed training and testing data from the models
folder, trains a Logistic Regression classifier, evaluates its performance,
and saves both the trained model and a confusion matrix visualization.
"""

import os
import logging
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_datasets(models_dir):
    """Load training and testing datasets from disk."""
    expected_files = {
        'X_train': 'X_train.pkl',
        'X_test': 'X_test.pkl',
        'y_train': 'y_train.pkl',
        'y_test': 'y_test.pkl',
    }

    datasets = {}
    for key, filename in expected_files.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

        logger.info(f"Loading {filename} from {models_dir}")
        datasets[key] = joblib.load(path)

    return datasets['X_train'], datasets['X_test'], datasets['y_train'], datasets['y_test']


def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression model."""
    logger.info("Initializing Logistic Regression model")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and return metrics and confusion matrix."""
    logger.info("Generating predictions for the test set")
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }

    return metrics


def save_confusion_matrix(confusion_mat, output_path):
    """Save the confusion matrix as a PNG image."""
    logger.info(f"Saving confusion matrix image to {output_path}")
    plt.figure(figsize=(6, 5))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Fake', 'True']
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = confusion_mat.max() / 2.0
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            plt.text(
                j,
                i,
                format(confusion_mat[i, j], 'd'),
                horizontalalignment='center',
                color='white' if confusion_mat[i, j] > thresh else 'black',
            )

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_trained_model(model, model_path):
    """Persist the trained model to disk."""
    logger.info(f"Saving trained model to {model_path}")
    joblib.dump(model, model_path)
    return model_path


def print_summary(training_count, testing_count, metrics, model_path):
    """Print the final training summary to the console."""
    summary_lines = [
        '\n' + '=' * 60,
        'MODEL TRAINING SUMMARY',
        '=' * 60,
        f'Training samples: {training_count}',
        f'Testing samples:  {testing_count}',
        f'Accuracy:         {metrics["accuracy"]:.4f}',
        f'Precision:        {metrics["precision"]:.4f}',
        f'Recall:           {metrics["recall"]:.4f}',
        f'F1 Score:         {metrics["f1_score"]:.4f}',
        f'Model saved path: {model_path}',
        '=' * 60,
        '\nClassification Report:\n',
        metrics['classification_report'],
    ]

    logger.info('\n'.join(summary_lines))
    print('\n'.join(summary_lines))


def main():
    """Run the model training and evaluation pipeline."""
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        models_dir = os.path.join(project_root, 'models')
        outputs_dir = os.path.join(project_root, 'outputs')

        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        os.makedirs(outputs_dir, exist_ok=True)
        logger.info(f"Ensured outputs folder exists: {outputs_dir}")

        X_train, X_test, y_train, y_test = load_datasets(models_dir)
        logger.info(f"Loaded datasets: X_train={X_train.shape}, X_test={X_test.shape}")

        model = train_logistic_regression(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        confusion_path = os.path.join(outputs_dir, 'confusion_matrix.png')
        save_confusion_matrix(metrics['confusion_matrix'], confusion_path)

        model_path = os.path.join(models_dir, 'fake_news_model.pkl')
        save_trained_model(model, model_path)

        print_summary(
            training_count=X_train.shape[0],
            testing_count=X_test.shape[0],
            metrics=metrics,
            model_path=model_path,
        )

        logger.info('Training pipeline completed successfully')

    except FileNotFoundError as err:
        logger.error(f'File not found: {err}')
        raise
    except Exception as err:
        logger.exception('Model training pipeline failed')
        raise


if __name__ == '__main__':
    main()
