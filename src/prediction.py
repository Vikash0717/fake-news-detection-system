"""
Prediction Module

This module loads a trained logistic regression model and TF-IDF vectorizer
from the models folder, predicts whether an input news article is real or fake,
and exposes a simple command-line interface.
"""

import os
import logging
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model_paths():
    """Return absolute paths for the model and vectorizer artifacts."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    models_dir = os.path.join(project_root, 'models')
    return (
        os.path.join(models_dir, 'fake_news_model.pkl'),
        os.path.join(models_dir, 'tfidf_vectorizer.pkl'),
    )


def load_artifacts():
    """Load the trained model and TF-IDF vectorizer from disk."""
    model_path, vectorizer_path = get_model_paths()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    logger.info('Loading trained model and TF-IDF vectorizer...')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    logger.info('Artifacts loaded successfully')

    return model, vectorizer


def validate_news_text(news_text):
    """Validate that the input text is a non-empty string."""
    if not isinstance(news_text, str):
        raise ValueError('News text must be a string')

    cleaned = news_text.strip()
    if not cleaned:
        raise ValueError('News text cannot be empty')

    return cleaned


def predict_news(news_text, model=None, vectorizer=None):
    """Predict whether a news article is real or fake."""
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    validated_text = validate_news_text(news_text)

    try:
        logger.info('Transforming input text using TF-IDF vectorizer')
        features = vectorizer.transform([validated_text])
    except Exception as exc:
        raise RuntimeError(f'Failed to vectorize input text: {exc}') from exc

    try:
        prediction = model.predict(features)[0]
    except Exception as exc:
        raise RuntimeError(f'Failed to predict news label: {exc}') from exc

    confidence = None
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))
            logger.info(f'Prediction confidence computed: {confidence:.4f}')
    except Exception:
        logger.warning('Confidence score unavailable for this model')
        confidence = None

    label = 'REAL NEWS' if int(prediction) == 1 else 'FAKE NEWS'
    return label, confidence


def print_prediction_result(label, confidence):
    """Print the prediction result and confidence score."""
    print('\nPrediction: {}'.format(label))
    if confidence is not None:
        print('Confidence Score: {:.4f}'.format(confidence))
    else:
        print('Confidence Score: Not available')


def run_example_tests(model, vectorizer):
    """Run two example predictions to verify the model behavior."""
    examples = [
        (
            'The government announced a new stimulus package to support small businesses and create jobs during the economic recovery.',
            'Likely Real News'
        ),
        (
            'Experts confirmed that the celebrity was found alive on a secret island and the entire story was hidden by the media.',
            'Likely Fake News'
        ),
    ]

    print('\nRunning example predictions...')
    for text, description in examples:
        try:
            label, confidence = predict_news(text, model=model, vectorizer=vectorizer)
            print('\nExample: ' + description)
            print('Article: ' + text)
            print_prediction_result(label, confidence)
        except Exception as exc:
            logger.error('Example prediction failed: %s', exc)
            print(f'Example prediction failed: {exc}')


def main():
    """Entry point for the prediction CLI."""
    try:
        model, vectorizer = load_artifacts()

        run_example_tests(model, vectorizer)

        print('\nEnter a news article to classify:')
        user_input = input().strip()
        if not user_input:
            raise ValueError('No input received. Please enter a news article.')

        label, confidence = predict_news(user_input, model=model, vectorizer=vectorizer)
        print_prediction_result(label, confidence)

    except FileNotFoundError as exc:
        logger.error('Required artifact missing: %s', exc)
        print(f'Error: {exc}')
    except ValueError as exc:
        logger.error('Invalid input: %s', exc)
        print(f'Input error: {exc}')
    except Exception as exc:
        logger.exception('Unexpected error during prediction')
        print(f'Unexpected error: {exc}')


if __name__ == '__main__':
    main()
