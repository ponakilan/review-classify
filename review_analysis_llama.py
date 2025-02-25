import json
import tqdm
import pandas as pd

drugs = [
    "Ocrevus Processed - ocrevus_reviews_processed_cleaned.csv_classified_reviews_multiple_models.csv",
    "Ocrelizumab Processed - ocrelizumab_reviews_processed_cleaned.csv_classified_reviews_multiple_models.csv"
]

for drug in drugs:
    df = pd.read_csv(drug)
    data = df.to_json(orient="records")
    
    from langchain_community.chat_models import ChatOllama
    
    model = ChatOllama(model="llama3.1:8b", temperature=0)
    
    classify_prompt = """
    Please classify the following review into one of the following categories: Positive, Negative, or Neutral.
    The output should be strictly in JSON format as shown below:
    {"class": "<Safety/Efficacy/Unmet needs/Access to diagnostics & treatment care/Lack of MS disease and symptom awareness/Higher time taken to reach a neurologist/Better alternative/Adherence/patient switchouts/Convenience>"}
    Only curly braces should prefix or sufix the above format. No other characters. A review should be classified into only one category.
    
    Review: {review}
    """
    
    model_response_list = []
    for i in tqdm.tqdm(range(len(data))):
        review = data[i]["key_points_llama"]
        prompt = classify_prompt.format(review=review)
    
        try:
            response = model.invoke(prompt)
            model_response = response.content.strip()
            model_response_list.append(model_response)
    
            try:
                classification = json.loads(model_response)
                if all(key in classification for key in ['class']):
                    data[i]["predicted_category_llama"] = classification["class"]
                else:
                    data[i]["predicted_category_llama"] = "Unknown"
    
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {model_response}")
                data[i]["predicted_category_llama"] = "Unknown"
    
        except Exception as e:
            print(f"Error classifying tweet: {e}")
            data[i]["predicted_category_llama"] = "Unknown"
    data = pd.read_json(data)
    data.to_csv(f"{drug}_classified_llama.csv", index=False)
