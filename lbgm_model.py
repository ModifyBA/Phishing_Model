import os #for handling file paths and directories
import json #for saving the threshold in a human-readable format
import pickle #for saving the feature names (since they are a list of strings, we can use pickle to save them as a binary file)
import joblib #for saving the model (joblib is more efficient for saving large models like LightGBM compared to pickle)
import numpy as np #for handling numerical operations and arrays
import pandas as pd #for handling dataframes and CSV files
import lightgbm as lgb 

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from FE import extract_features

#just some file paths for loading/saving data and model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "clean_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "clean_test.csv")
VALIDATION_PATH = os.path.join(BASE_DIR, "clean_valid.csv")

MODEL_PATH = os.path.join(BASE_DIR, "lbgm_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.json")


#label cleaning, just in case
PHISHING_LABELS = {"phishing", "malicious", "bad", "1", "phish", "true", "yes"}
BENIGN_LABELS = {"benign", "good", "legitimate", "0", "clean", "false", "no"}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy() #makes a copy of the dataframe to avoid modifying the original

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame must contain 'url' and 'label' columns")
    
    df["url"] = df["url"].astype(str).str.strip() #just in case there are any leading/trailing spaces in the URLs
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    valid_mask = df["label"].isin(PHISHING_LABELS.union(BENIGN_LABELS))
    df = df[valid_mask].copy()

    df["label_num"] = df["label"].apply(lambda x: 1 if x in PHISHING_LABELS else 0) #converts the label to a binary format (1 for phishing, 0 for benign)   

    df = df[df["url"] != ""].copy()
    df = df.drop_duplicates(subset=["url"])

    return df

#puts it into a python type spreadsheet and makes it easier to manipulate
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
valid_df = pd.read_csv(VALIDATION_PATH)

train_df = clean_dataset(train_df) #cleans the dataset in the training dataframe by applying the clean_dataset function, which standardizes the labels to a binary format and removes any invalid entries, ensuring that the training data is in a consistent format for model training
test_df = clean_dataset(test_df)
valid_df = clean_dataset(valid_df)

print("Train size:", len(train_df))
print("Valid size:", len(valid_df))
print("Test size:", len(test_df))

print("Train class counts:")
print(train_df["label_num"].value_counts())

def build_feature_matrix(urls: list) -> pd.DataFrame: #takes in a list of urls and returns a feature matrix (pandas dataframe) that can be fed into the model
    rows = []
    for url in urls:
        try:
            feats = extract_features(url)
            rows.append(feats)
        except Exception as e:
            print(f"Error occurred while extracting features for URL: {url}")
            print(f"Error message: {e}")
            rows.append({}) #append an empty dict for URLs that cause errors, which will be filled with 0s later

    X = pd.DataFrame(rows)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0) #handle any infinite or NaN values that might arise from feature extraction
    X = X.select_dtypes(include=[np.number]) #keep only numeric columns, since the model can only handle numeric features
    return X

X_train = build_feature_matrix(train_df["url"].tolist()) #builds the feature matrix for the training data by applying the feature extraction function to each URL in the training dataframe, resulting in a new dataframe where each row corresponds to a URL and each column corresponds to a feature extracted from that URL
X_valid = build_feature_matrix(valid_df["url"].tolist())
X_test = build_feature_matrix(test_df["url"].tolist())

y_train = train_df["label_num"].to_numpy() #extracts the labels from the training dataframe and converts them to a numpy array
y_valid = valid_df["label_num"].to_numpy()  
y_test = test_df["label_num"].to_numpy()

feature_columns = X_train.columns.tolist() #stores the feature column names for later use (e.g., when reindexing the validation and test feature matrices to ensure they have the same columns as the training feature matrix)

X_valid = X_valid.reindex(columns=feature_columns, fill_value=0) #reindexing the validation and test feature matrices to ensure they have the same columns as the training feature matrix, filling any missing columns with 0s (in case some features are not present in the validation/test sets)
X_test = X_test.reindex(columns=feature_columns, fill_value=0)

neg_count = int((y_train == 0).sum()) #counts the number of benign samples in the training data
pos_count = int((y_train == 1).sum()) #counts the number of phishing samples in the training data
scale_pos_weight = neg_count / max(1, pos_count) #calculates the scale_pos_weight for LightGBM, which is the ratio of negative samples to positive samples, used to handle class imbalance

model = lgb.LGBMClassifier( #initializes the LightGBM model with specific hyperparameters
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.0,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=-1)

model.fit(X_train, y_train, 
          eval_set=[(X_valid, y_valid)], 
          eval_metric="auc", 
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=100)],
          ) #trains the model on the training data

valid_probs = model.predict_proba(X_valid)[:, 1] #gets the predicted probabilities for the validation set (the probability of being in the positive class, i.e., phishing)
test_probs = model.predict_proba(X_test)[:, 1] #gets the predicted probabilities for the test set

def evaluate_threshold(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int) #converts the predicted probabilities into binary predictions based on the threshold (1 for phishing, 0 for benign)  
    
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel() #calculates the confusion matrix and unpacks it into true negatives, false positives, false negatives, and true positives

    precision = precision_score(y_true, preds, zero_division=0) #calculates the precision of the predictions
    recall = recall_score(y_true, preds, zero_division=0) #calculates the recall of the predictions
    f1 = f1_score(y_true, preds, zero_division=0) #calculates the F1 score of the predictions
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 #calculates the false positive rate of the predictions
    return{
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def choose_threshold_with_fpr_budget(y_true, probs, target_fpr=0.01):
    best_score = None
    best_stats = None

    for t in np.arange(0.10, 0.96, 0.01):
        stats = evaluate_threshold(y_true, probs, float(t))

        if stats["fpr"] <= target_fpr:
            score = (stats["recall"], stats["precision"], stats["f1"], -stats["threshold"]) #prioritize recall, then precision, then F1 score, and finally prefer lower thresholds (more aggressive detection) if all else is equal 
            if best_score is None or score > best_score:
                best_score = score
                best_stats = stats

    if best_stats is None:
        for t in np.arange(0.10, 0.96, 0.01):
            stats = evaluate_threshold(y_true, probs, float(t))
            score = (stats["f1"], -stats["fpr"], stats["recall"])
            if best_score is None or score > best_score:
                best_score = score
                best_stats = stats

    return best_stats

threshold_stats = choose_threshold_with_fpr_budget(y_valid, valid_probs, target_fpr=0.01)
best_threshold = threshold_stats["threshold"]
test_preds = (test_probs >= best_threshold).astype(int) #converts the predicted probabilities into binary predictions based on the threshold (1 for phishing, 0 for benign)  

print("Best threshold:", best_threshold)
print("Validation ROC-AUC:", roc_auc_score(y_valid, valid_probs))
print("Validation PR-AUC :", average_precision_score(y_valid, valid_probs))

print("Chosen threshold:")
print(json.dumps(threshold_stats, indent=2))

print("Test ROC-AUC:", roc_auc_score(y_test, test_probs))
print("Test PR-AUC :", average_precision_score(y_test, test_probs))

print("Classification Report:")
print(classification_report(y_test, test_preds, digits=4, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_preds))
# Save the model, feature names, and threshold for later use
importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 15 Features:")
print(importance_df.head(15).to_string(index=False))

joblib.dump(model, MODEL_PATH)

with open(FEATURES_PATH, "wb") as f:
    pickle.dump(feature_columns, f)

with open(THRESHOLD_PATH, "w") as f:
    json.dump(
        {
            "threshold": best_threshold,
            "validation_stats": threshold_stats
        },
        f,
        indent=2
    )

