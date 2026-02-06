import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re 
import string
import optuna
import imblearn
string.punctuation
import contractions
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import LinearSVC 
from sklearn.metrics import classification_report
#nltk.download('stopwords')
#nltk.download('punkt') 
#nltk.download('punkt_tab')
#nltk.download('wordnet')
#Load Data
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Inno\reviews_badminton\data.csv")
data = df.copy()
#print(df.head())
#print(df.columns)
#print(df.info())




def clean(doc, stem=False):
    doc = contractions.fix(str(doc))
    doc = re.sub(r'[^a-zA-Z]', ' ', doc)
    doc = doc.lower()

    tokens = word_tokenize(doc)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]

    return " ".join(tokens)






data['cleaned_review'] = data['Review text'].apply(clean)
#print(data[['Review text','cleaned_review']].head())


def convert_rating(r):
    if r <= 2:
        return "negative" 
    elif r == 3:
        return "neutral" 
    else: return "positive"
data['Sentiment'] = data['Ratings'].apply(convert_rating)
#print(data[['Ratings', 'Sentiment']].head())



"""#stemming
stemmer = PorterStemmer()
def stem_text(text):
    words = nltk.word_tokenize(str(text)) 
    return " ".join([stemmer.stem(w) for w in words])


#lemmatizing
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = nltk.word_tokenize(str(text)) 
    return " ".join([lemmatizer.lemmatize(w) for w in words])
data['normalized_review'] = data['cleaned_review'].apply(lemmatize_text)
#print(data[['cleaned_review', 'normalized_review']].head())"""

data = data[['cleaned_review', 'Sentiment']].dropna()
data = data.reset_index(drop=True)

tfidf = TfidfVectorizer(max_features=500,ngram_range = (1, 3))
X = tfidf.fit_transform(data['cleaned_review'])
y = data['Sentiment']
#print(tfidf_df.head())
#tfidf_df.to_csv("tfidf_features.csv", index=False)


#Model 
from sklearn.model_selection import train_test_split
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

#print("Before:", y.value_counts())
#print("After:", pd.Series(y_res).value_counts())

X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.2,stratify=y_res,random_state=42)
#print(X_train.shape,"\n", X_test.shape,"\n", y_train.shape,"\n", y_test.shape)




import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

# Make sure X_train, X_test, y_train, y_test are already defined


def objective(trial):

    model_name = trial.suggest_categorical("model", ["logreg", "nb", "svm"])

    # Logistic Regression
    if model_name == "logreg":
        C = trial.suggest_float("lr_C", 1e-4, 10.0, log=True)
        solver = trial.suggest_categorical("lr_solver", ["lbfgs", "newton-cg", "saga"])

        class_weight = trial.suggest_categorical("lr_class_weight", [None, "balanced"])


        clf = LogisticRegression(
            C=C,
            solver=solver,
            penalty="l2",
            class_weight=class_weight,

            )


 

        

    # Multinomial Naive Bayes
    elif model_name == "nb":
        alpha = trial.suggest_float("nb_alpha", 1e-3, 10.0, log=True)
        fit_prior = trial.suggest_categorical("nb_fit_prior", [True, False])

        clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

    # Linear SVM
    else:
        C = trial.suggest_float("svm_C", 1e-4, 10.0, log=True)
        class_weight = trial.suggest_categorical("svm_class_weight", [None, "balanced"])

        clf = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=5000
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy").mean()

    return score


study = optuna.create_study(direction="maximize", study_name="sentiment_analysis")
study.optimize(objective, n_trials=50)

print("\nBest Hyperparameters:")
print(study.best_params)



best_params = study.best_params
model_name = best_params["model"]

if model_name == "logreg":
    best_model = LogisticRegression(
        C=best_params["lr_C"],
        solver=best_params["lr_solver"],
        penalty="l2",
        class_weight=best_params["lr_class_weight"],
        max_iter=1000
    )

elif model_name == "nb":
    best_model = MultinomialNB(
        alpha=best_params["nb_alpha"],
        fit_prior=best_params["nb_fit_prior"]
    )

else:
    best_model = LinearSVC(
        C=best_params["svm_C"],
        class_weight=best_params["svm_class_weight"],
        max_iter=5000
    )

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"\nBest Model Selected by Optuna: {model_name}")
print(classification_report(y_test, y_pred))



import pickle

# Save model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved!")










