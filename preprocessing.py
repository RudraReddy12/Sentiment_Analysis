# Importing Required Libraries 
import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle

# Load Data
def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(path)

# EDA
def perform_eda(df: pd.DataFrame) -> None:
    """Perform basic EDA and plot ratings distribution."""

    # Print dataset info (columns, datatypes, null values)
    print("Data Info:")
    print(df.info())

    # Show first 5 rows
    print("\nFirst 5 rows:")
    print(df.head())

    # Show distribution of ratings
    print("\nRatings distribution:")
    print(df['Ratings'].value_counts())

    # Plot ratings distribution
    sns.countplot(x='Ratings', data=df)
    plt.title("Ratings Distribution")
    plt.show()

# Preprocessing Utilities
# Define stopwords but keep negations (important for sentiment analysis)
stop_words = set(stopwords.words('english'))
negations = {"not", "no", "nor", "never"}
stop_words = stop_words - negations

def clean_text(text: str) -> str:
    """Remove non-alphabetic chars, lowercase, and remove stopwords."""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)) # Remove special characters/numbers
    words = text.lower().split() # Convert to lowercase and split

    words = [w for w in words if w not in stop_words] # Remove stopwords (except negations)
    return " ".join(words)

def convert_rating(r: int) -> str:
    """Convert numeric rating to sentiment label."""
    if r <= 2:
        return "negative"
    elif r == 3:
        return "neutral"
    else:
        return "positive"

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text: str) -> str:
    """Lemmatize words in text."""
    words = nltk.word_tokenize(str(text)) # Tokenize text into words
    return " ".join([lemmatizer.lemmatize(w) for w in words])

def positive_word_count(texts: pd.Series) -> np.ndarray:
    """Count positive words in each text."""
    pos_words = {"excited", "amazing", "great", "love", "happy"}

    # For each text, count how many positive words appear
    return np.array([[sum(word in pos_words for word in t.split())] for t in texts])

# Preprocess Data
def preprocess_data(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    1. Clean and normalize text
    2. Convert ratings to sentiment labels
    3. Apply lemmatization
    4. Vectorize text using TF-IDF
    5. Add positivity word count feature
    6. Balance dataset using SMOTE
    7. Split into train/test sets
    8. Save TF-IDF vectorizer for later use
    """
    # Step 1: Clean text
    df['cleaned_review'] = df['Review text'].apply(clean_text)

    # Step 2: Convert ratings to sentiment labels
    df['Sentiment'] = df['Ratings'].apply(convert_rating)

    # Step 3: Lemmatize text
    df['normalized_review'] = df['cleaned_review'].apply(lemmatize_text)

    # Step 4: TF-IDF vectorization (unigrams + bigrams, max 10k features)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['normalized_review'].astype(str))

    # Step 5: Add positivity word count feature
    X_pos = positive_word_count(df['normalized_review'])

    # Step 6: Combine TF-IDF features with positivity count
    X_combined = np.hstack([X_tfidf.toarray(), X_pos])
    feature_names = list(tfidf.get_feature_names_out()) + ["positive_word_count"]
    X = pd.DataFrame(X_combined, columns=feature_names)

    # Target variable
    y = df['Sentiment']

    # Step 7: Balance dataset using SMOTE (oversampling minority classes)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Step 8: Train-test split (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    # Step 9: Save TF-IDF vectorizer for future inference
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    return X_train, X_test, y_train, y_test
