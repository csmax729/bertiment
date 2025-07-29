import streamlit as st
from transformers import pipeline

# Use caching to load the model only once
@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# --- App Layout ---
st.title('🤖 Real-Time Product Review Sentiment Analysis')
st.write(
    "Enter a product review below to see whether its sentiment is positive or negative. "
    "This app uses a pre-trained DistilBERT model from Hugging Face."
)

# Load the model
sentiment_pipeline = load_model()

# User input
user_input = st.text_area("Enter the review text here:", "I loved this product, it's the best purchase I've made all year!")

# Analyze button
if st.button('Analyze Sentiment'):
    if user_input:
        # Get the prediction
        result = sentiment_pipeline(user_input)
        label = result[0]['label']
        score = result[0]['score']

        # Display the result with an icon
        if label == 'POSITIVE':
            st.success(f'Sentiment: {label} (Confidence: {score:.2f}) 👍')
        else:
            st.error(f'Sentiment: {label} (Confidence: {score:.2f}) 👎')
    else:
        st.warning('Please enter a review to analyze.')