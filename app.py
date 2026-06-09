"""
Fake News Detection System

Streamlit web application for classifying news articles as real or fake.
Uses a trained Logistic Regression model and TF-IDF vectorization.
"""

import os
import logging
import joblib
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_artifact_paths():
    """Return absolute paths for the model artifacts."""
    root_dir = os.path.dirname(__file__)
    models_dir = os.path.join(root_dir, 'models')
    return (
        os.path.join(models_dir, 'fake_news_model.pkl'),
        os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    )


def load_artifacts():
    """Load the trained model and TF-IDF vectorizer."""
    model_path, vectorizer_path = get_artifact_paths()

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model artifact not found: {model_path}')
    if not os.path.isfile(vectorizer_path):
        raise FileNotFoundError(f'Vectorizer artifact not found: {vectorizer_path}')

    logger.info('Loading model and vectorizer artifacts')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def validate_text(news_text):
    """Validate the news text input."""
    if not isinstance(news_text, str):
        raise ValueError('Please enter valid text.')

    cleaned_text = news_text.strip()
    if not cleaned_text:
        raise ValueError('News article text cannot be empty.')

    return cleaned_text


def predict_news(news_text, model, vectorizer):
    """Predict whether the given news text is real or fake."""
    text = validate_text(news_text)
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities)) * 100.0

    label = 'REAL NEWS' if int(prediction) == 1 else 'FAKE NEWS'
    return label, confidence


def display_sidebar():
    """Render the sidebar with project details."""
    st.sidebar.title('Project Information')
    st.sidebar.markdown('### Fake News Detection System')
    st.sidebar.markdown('This app classifies news articles as real or fake using a trained Logistic Regression model.')
    st.sidebar.divider()
    st.sidebar.markdown('**Model Accuracy**')
    st.sidebar.write('98.93%')
    st.sidebar.markdown('**Dataset Size**')
    st.sidebar.write('39,100 records')
    st.sidebar.divider()
    st.sidebar.markdown('### Technology Stack')
    st.sidebar.markdown(
        """
- Python  
- Pandas  
- Scikit-Learn  
- TF-IDF  
- Logistic Regression  
- Streamlit
        """
    )
    st.sidebar.divider()
    st.sidebar.markdown('### Usage')
    st.sidebar.write(
        """
1. Paste a news article  
2. Click Predict  
3. Review the result and confidence score
        """
    )


def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title='Fake News Detection System',
        page_icon='📰',
        layout='centered',
        initial_sidebar_state='expanded'
    )

    display_sidebar()

    st.title('Fake News Detection System')
    st.markdown(
        'Use this application to classify news articles as real or fake using a Logistic Regression model trained on TF-IDF features.'
    )

    st.divider()

    news_text = st.text_area(
        'Enter news article text',
        placeholder='Paste the news article text here...',
        height=300
    )

    if st.button('Predict'):
        try:
            model, vectorizer = load_artifacts()
            label, confidence = predict_news(news_text, model, vectorizer)

            if label == 'REAL NEWS':
                st.success(f'Prediction: {label}')
            else:
                st.error(f'Prediction: {label}')

            if confidence is not None:
                st.metric('Confidence Score', f'{confidence:.2f}%')
            else:
                st.warning('Confidence score is unavailable for this model.')

        except FileNotFoundError as exc:
            st.error(f'Artifact error: {exc}')
            logger.error('Artifact error: %s', exc)
        except ValueError as exc:
            st.warning(str(exc))
            logger.warning('Validation error: %s', exc)
        except Exception as exc:
            st.error('An unexpected error occurred during prediction.')
            logger.exception('Prediction failed: %s', exc)

    st.write('')
    st.write('---')
    st.header('Prediction Output')
    st.write('Enter a news article and click Predict to see the classification result.')


if __name__ == '__main__':
    main()
