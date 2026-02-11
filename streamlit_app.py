import os
import re

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem import WordNetLemmatizer

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_dir)

nltk.download("wordnet", download_dir=nltk_data_dir)
nltk.download("omw-1.4", download_dir=nltk_data_dir)
# Load Model
@st.cache_resource
def load_model():
    return joblib.load('sentiment_mnb_model.pkl')

model = load_model()

st.title("What's the Review: Sentiment Analysis")
st.markdown("Enter a **Review** of your choice and it will be output as **Positive** or **Negative** review.")

review = st.text_input("Enter a review")

def clean(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = text.replace('<br /><br />', ' ') # Replace HTML linebreak format with empty space
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    # Tokenize and lemmatize each word
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

if st.button("Analyze Review", type = "primary"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        clean_review = clean(review)
        pred = model.predict([clean_review])[0]
    
        if pred == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")