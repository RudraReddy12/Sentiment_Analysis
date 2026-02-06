import streamlit as st
import pickle
import re
import string
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download once (Streamlit cloud safe)
nltk.download('punkt')
nltk.download('stopwords')

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------------- TEXT CLEANING ----------------
def clean_text(doc):
    doc = contractions.fix(str(doc))
    doc = re.sub(r'[^a-zA-Z]', ' ', doc)
    doc = doc.lower()

    tokens = word_tokenize(doc)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]

    return " ".join(tokens)

# ---------------- STREAMLIT UI ----------------
st.title("Badminton Review Sentiment Analyzer")
st.write("This model predicts whether a review is **Positive, Neutral, or Negative**.")

user_input = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == "positive":
            st.success(" Positive Review")
        elif prediction == "neutral":
            st.info(" Neutral Review")
        else:
            st.error(" Negative Review")



