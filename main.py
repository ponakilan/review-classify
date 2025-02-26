from datasets import load_dataset
from transformers import pipeline
import pandas as pd

files = [
    "to QC drugs.com MS data - Sheet1.csv"
]

for file in files:
    dataset = load_dataset("csv", data_files=file)["train"]
    
    # Define the candidate labels for classification
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
    
    # Define a dictionary of zero-shot classification models to evaluate
    models = {
        "bart": "facebook/bart-large-mnli",
        "deberta": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c",
        "roberta": "MoritzLaurer/roberta-large-zeroshot-v2.0-c",
        "distilbart": "valhalla/distilbart-mnli-12-1"
    }
    
    # Define a function to classify a single example with the given model pipeline
    def classify_with_model(example, model_name, classifier):
        # Use the "comment" field as the input text
        result = classifier(example["key_points_llama"], candidate_labels=categories)
        # Save the top predicted category and all scores to new columns
        example[f"predicted_category_{model_name}"] = result["labels"][0]
        example[f"scores_{model_name}"] = result["scores"]
        return example
    
    # Loop over each model, process the dataset, and add new classification columns
    for model_name, model_checkpoint in models.items():
        print(f"Processing model: {model_name}")
        classifier = pipeline("zero-shot-classification", model=model_checkpoint)
        dataset = dataset.map(lambda x: classify_with_model(x, model_name, classifier))
    
    # Preview the first few examples
    print(dataset[:5])
    
    # Optionally, convert the classified dataset to a Pandas DataFrame and save the results as a CSV file
    df = pd.DataFrame(dataset)
    df.to_csv(f"{file}_classified_reviews_multiple_models.csv", index=False)
