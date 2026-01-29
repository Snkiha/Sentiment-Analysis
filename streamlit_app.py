import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Load Model
@st.cache_resource
def load_model():
    return joblib.load('sentiment_mnb_model.pkl')

model = load_model()

st.title("What's the Review: Sentiment Analysis")
st.markdown("Enter a **Review** of your choice and it will be output as ** Positive** or **Negative** review.")