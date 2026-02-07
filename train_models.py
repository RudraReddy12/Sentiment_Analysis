# Importing Required Libraries
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocessing import load_data, perform_eda, preprocess_data

def train_models(X_train, y_train):
    # Train multiple ml models on the training dataset.
    # Saves each trained model as a pickle file for later use.
    models = {}

    # Logistic Regression with balanced class weights to handle class imbalance
    lr = LogisticRegression(max_iter=5000, class_weight='balanced')
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

    # Multinomial Naive Bayes (commonly used for text classification)
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    models["NaiveBayes"] = nb

    # Linear Support Vector Machine with custom class weights 
    # Giving more importance to the 'positive' class
    svm = LinearSVC(class_weight={'negative': 1, 'neutral': 1, 'positive': 2}, max_iter=5000) 
    svm.fit(X_train, y_train)
    models["SVM"] = svm

    # Save all models individually
    for name, model in models.items():
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    return models

def evaluate_models(models, X_test, y_test):

    """
    Evaluate trained models on the test dataset.
    Prints performance metrics and selects the best model based on F1 score.
    Saves the best model separately as 'best_model.pkl'.
    """
    scores = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")

        # Store metrics in dictionary
        scores[name] = {
            "accuracy": acc,
            "f1_macro": f1,
            "precision_macro": precision,
            "recall_macro": recall
        }

        # Print model performance
        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Pick best model by F1 score
    best_model_name = max(scores, key=lambda k: scores[k]["f1_macro"])
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name} "
          f"(F1: {scores[best_model_name]['f1_macro']:.4f})")

    # Save best model separately
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model_name, best_model, scores

if __name__ == "__main__":

    """ Main execution block: 
        1. Load dataset 
        2. Perform exploratory data analysis (EDA) 
        3. Preprocess data into train/test sets 
        4. Train models 
        5. Evaluate models and save the best one """ 
    
    # Load dataset from CSV file
    df = load_data(r"C:\Users\ragha\python_files\innomatics\sentiment_analysis_datasets\reviews_badminton\data.csv")
    
    # Perform exploratory data analysis (visualizations, statistics, etc.)
    perform_eda(df)

    # Preprocess data (cleaning, feature extraction, splitting into train/test sets)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models and save the best one
    best_model_name, best_model, scores = evaluate_models(models, X_test, y_test)
