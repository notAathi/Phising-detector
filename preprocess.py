import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)  
    text = re.sub(r"\W", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()  
    text = " ".join([word for word in text.lower().split() if word not in stop_words])
    return text


df = pd.read_csv("website_legitimacy_data.csv")

if "html" in df.columns:
    df["cleaned_text"] = df["html"].astype(str).apply(clean_text)
else:
    raise ValueError("Column 'html' not found in dataset.")

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df["cleaned_text"])

df.to_csv("processed_data.csv", index=False)
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Data preprocessing complete! Saved as processed_data.csv")
