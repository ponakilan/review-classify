from transformers import pipeline
import pandas as pd
from tqdm import tqdm

# Load your CSV file (ensure the file has a 'review' column)
df = pd.read_csv('Drugs.com.csv')
reviews = df['comment'].tolist()

# Define your categories
categories = [
    "Safety",
    "Efficacy",
    "Unmet needs",
    "Access to diagnostics & treatment care",
    "Lack of MS disease and symptom awareness",
    "Higher time taken to reach a neurologist",
    "Better alternative",
    "Adherence/patient switchouts",
    "Convenience"
]

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Perform zero-shot classification
results = []
for review in tqdm(reviews):
    result = classifier(review, candidate_labels=categories)
    results.append({
        "review": review,
        "predicted_category": result['labels'][0],  # Top predicted category
        "scores": result['scores']                 # Scores for all categories
    })

results_df = pd.DataFrame(results)
print(results_df.head())

results_df.to_csv('classified_reviews.csv', index=False)
