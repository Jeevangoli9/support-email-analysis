# Directory structure (copy these into your repo)
# support-email-analysis/
# ├── data/
# │   └── Sample_Support_Emails_Dataset.csv
# ├── src/
# │   └── email_analysis.py
# ├── requirements.txt
# └── README.md

# ============================
# requirements.txt
# ============================
pandas
scikit-learn
nltk
matplotlib

# ============================
# README.md
# ============================
"""
# Support Email Analysis

## Overview
This project analyzes customer support emails to identify common issues, categorize queries, and generate insights.

## Dataset
- 20 sample emails with sender, subject, body, and sent date.

## Features
- Email categorization (Billing, Login, API, Subscription, General)
- Frequency analysis by sender and date
- Keyword extraction from subject/body
- Visualization of top issues

## How to Run
1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd support-email-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   ```bash
   python src/email_analysis.py
   ```

## Tools Used
- Python
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
"""

# ============================
# src/email_analysis.py
# ============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if not already present
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/Sample_Support_Emails_Dataset.csv")

# Function to categorize emails
def categorize_email(text):
    text = text.lower()
    if "billing" in text or "pricing" in text:
        return "Billing"
    elif "login" in text or "password" in text or "access" in text:
        return "Login Issue"
    elif "api" in text or "integration" in text:
        return "API"
    elif "subscription" in text:
        return "Subscription"
    else:
        return "General"

# Apply categorization
df["category"] = df["subject"].apply(categorize_email)

# Frequency analysis
print("Emails per category:\n", df["category"].value_counts())
print("\nEmails per sender:\n", df["sender"].value_counts())

# Keyword extraction from subjects
vectorizer = CountVectorizer(stop_words=stopwords.words("english"))
X = vectorizer.fit_transform(df["subject"])
keywords = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False)

print("\nTop keywords in subjects:\n", keywords.head(10))

# Visualization of categories
plt.figure(figsize=(6,4))
df["category"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Email Categories")
plt.ylabel("Count")
plt.show()
