import requests
import json
import time
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv
import os



load_dotenv()

serp_key = os.getenv("SERP_API_KEY")

websites = ["Amazon", "Adidas", "Puma", "Paypal", "Flipkart", "Netflix"]
fake_websites = ["Amozoan", "Abibas", "Poma", "Paypel", "Flpkart", "Netfliix"]

def get_html_from_google(query, num_results=5):
    """Fetch HTML content of top search results using SerpAPI."""
    url = f"https://serpapi.com/search.json?q={query}&api_key={serp_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"⚠️ Failed to fetch results for: {query}")
        return []
    
    data = response.json()
    html_data = []
    
    for result in data.get("organic_results", [])[:num_results]:  
        link = result.get("link")
        try:
            page_response = requests.get(link, timeout=5)
            # soup = BeautifulSoup(page_response.text, "html.parser")
            html_data.append({"query": query, "url": link, "html": page_response.text.replace("\\n", "")})
            print(f"✅ Scraped: {link}")
        except:
            print(f"⚠️ Skipped: {link}")
    
    return html_data

real_data, fake_data = [], []

for site in websites:
    real_data.extend(get_html_from_google(f"Is {site} legit?", num_results=5))
    time.sleep(2) 

for site in fake_websites:
    fake_data.extend(get_html_from_google(f"Is {site} legit?", num_results=5))
    time.sleep(2)

df_real = pd.DataFrame(real_data)
df_fake = pd.DataFrame(fake_data)

df_real["label"] = 1  
df_fake["label"] = 0  

df = pd.concat([df_real, df_fake], ignore_index=True)
df.to_csv("website_legitimacy_data.csv", index=False)

print("\n✅ Data saved as website_legitimacy_data.csv")
