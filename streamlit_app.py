import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Load Model
@st.cache_resource
def load_model():
    return joblib.load('sentiment_mnb_model.pkl')

model = load_model()
