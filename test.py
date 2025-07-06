import os
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_FILE = "scraped_data.csv"

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=['site', 'text', 'label'])

site_labels = {row['site']: row['label'] for _, row in df.iterrows() if not pd.isna(row['label'])}

def scrape_data(site):
    """Simulates scraping and returns dummy text data"""
    print(f"✅ Scraped 10 results for {site}.")
    return "Sample scraped text content for " + site

def train_and_save_model(df):
    """Trains the model and saves it"""
    df = df.dropna(subset=['label'])  
    df['label'] = df['label'].astype(int) 

    if df.empty:
        print("⚠️ Not enough data to train the model.")
        return

    X = df.drop(columns=['site', 'label']) 
    y = df['label']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    model.save_model("model.json")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Model trained with accuracy: {acc:.2f}")

def scrape_train_and_predict(site):
    """Scrapes the site, stores data, and predicts"""
    global df

    if site in site_labels:
        print(f"📌 {site} is already labeled as {'real' if site_labels[site] == 0 else 'fake'}.\n")
        return 

    label = input(f"📌 Is this a real or fake site? (real/fake): ").strip().lower()
    label = 0 if label == "real" else 1  

    site_labels[site] = label 

    text_data = scrape_data(site)

    new_entry = pd.DataFrame([[site, text_data, label]], columns=['site', 'text', 'label'])
    df = pd.concat([df, new_entry], ignore_index=True)

    # Save panna marandhiraadha da LOL. so yeah the updated dataset olunga pannidu
    df.to_csv(DATA_FILE, index=False)

    train_and_save_model(df)  

while True:
    site = input("\n🔍 Enter a website name to check (or type 'exit' to stop): ").strip().lower()

    if site == "exit":
        break

    scrape_train_and_predict(site)
