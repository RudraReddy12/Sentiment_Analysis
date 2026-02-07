# importing required libraries
import streamlit as st
import pickle
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved model and vectorizer
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Preprocessing utilities
stop_words = set(stopwords.words('english'))
negations = {"not", "no", "nor", "never"}
stop_words = stop_words - negations
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    words = text.lower().split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def lemmatize_text(text: str) -> str:
    words = nltk.word_tokenize(str(text))
    return " ".join([lemmatizer.lemmatize(w) for w in words])

def positive_word_count(text: str) -> int:
    pos_words = {"excited", "amazing", "great", "love", "happy"}
    return sum(word in pos_words for word in text.split())

# Streamlit App
st.title("Sentiment Analysis on Badminton Reviews")
st.write("Enter a review and get sentiment prediction (positive, neutral, negative).")

user_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess
        cleaned = clean_text(user_input)
        normalized = lemmatize_text(cleaned)

        # TF-IDF transform
        X_tfidf = tfidf.transform([normalized]).toarray()

        # Positivity feature
        X_pos = np.array([[positive_word_count(normalized)]])

        # Combine features
        X_combined = np.hstack([X_tfidf, X_pos])

        # Predict
        prediction = model.predict(X_combined)[0]

        # Display result
        st.success(f"Predicted Sentiment: **{prediction.upper()}**")

        # Optional: Add emoji feedback
        if prediction == "positive":
            st.markdown("This looks like a happy review!")
        elif prediction == "negative":
            st.markdown("This seems to be a negative review.")
        else:
            st.markdown("This review is neutral.")
