import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# Load the CSV file
data = pd.read_csv("emails.csv")

# Check that the required columns exist
if "text" not in data.columns or "label" not in data.columns:
    raise ValueError('emails.csv must contain columns named "text" and "label".')

X = data["text"].fillna("").astype(str)
raw_labels = data["label"].fillna("").astype(str).str.strip().str.lower()

# Convert common label names to the two labels used by the model
label_map = {
    "spam": "spam",
    "ham": "not spam",
    "not spam": "not spam",
    "legitimate": "not spam",
}

y = raw_labels.map(label_map)

if y.isna().any():
    unknown = raw_labels[y.isna()].unique()
    raise ValueError(f"Unknown labels in emails.csv: {unknown}")

if y.nunique() < 2:
    raise ValueError("The CSV must include both spam and not-spam emails.")

# Create and train the model using all rows in the CSV
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )),
    ("classifier", MultinomialNB(alpha=0.5)),
])

model.fit(X, y)

# Save the trained model
joblib.dump(model, "spam_model.joblib")
print("Model saved as spam_model.joblib")

# Enter an email to classify
new_email = input("Enter an email to classify: ")
prediction = model.predict([new_email])[0]

print(f"Prediction: {prediction}")