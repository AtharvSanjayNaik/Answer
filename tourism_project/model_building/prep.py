# for data manipulation
import pandas as pd
# for creating folders
import os
# for splitting the data
from sklearn.model_selection import train_test_split
# for hugging face hub authentication to upload files
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/ASNaik/tourism-wellness-package/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop identifier columns that carry no predictive signal
df.drop(columns=[col for col in ["Unnamed: 0", "CustomerID"] if col in df.columns], inplace=True)

# Fix inconsistent category labels found in the raw data
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

target_col = "ProdTaken"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform a stratified train-test split (target classes are imbalanced)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="<-------Hugging Face Username------->/tourism-wellness-package",
        repo_type="dataset",
    )
print("Train/test splits uploaded to Hugging Face Hub.")
