import kagglehub
import pandas as pd
kagglehub.login()

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

ponakilan_llama_processed_reviews_path = kagglehub.dataset_download('ponakilan/llama-processed-reviews')
ponakilan_processed_json_path = kagglehub.dataset_download('ponakilan/processed-json')
ponakilan_llama_all_path = kagglehub.dataset_download('ponakilan/llama-all')
keras_gemma_keras_gemma_instruct_2b_en_2_path = kagglehub.model_download('keras/gemma/Keras/gemma_instruct_2b_en/2')

print('Data source import complete.')

# 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

from huggingface_hub import login
import json

login(token="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")

drugs = [
    "Ocrevus Processed - ocrevus_reviews_processed_cleaned.csv_classified_reviews_multiple_models.csv_classified_llama.csv",
    "Ocrelizumab Processed - ocrelizumab_reviews_processed_cleaned.csv_classified_reviews_multiple_models.csv_classified_llama.csv"
]

for drug in drugs:
    df = pd.read_csv(drug)
    data = json.loads(df.to_json(orient="records"))
    
    import os
    
    os.environ["KAGGLE_USERNAME"] = "ponakilan"
    os.environ["KAGGLE_KEY"] = "1d85627ef8bb7b294779dcdb5735aebf"
    
    os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch".
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    
    import keras_hub
    
    gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_instruct_2b_en")
    
    gemma_lm.generate("hello")
    
    import tqdm
    import json
    
    classify_prompt = """
    Please classify the following review into one of the following categories.
    The output should be strictly in JSON format as shown below:
    "class": "<Safety/Efficacy/Unmet needs/Access to diagnostics & treatment care/Lack of MS disease and symptom awareness/Higher time taken to reach a neurologist/Better alternative/Adherence/patient switchouts/Convenience>"
    Only curly braces should prefix or sufix the above format. No other characters. A review should be classified into only one category.
    
    Review: {review}
    
    Strictly follow the json format. JSON format is very very important.
    """
    
    model_response_list = []
    for i in tqdm.tqdm(range(len(data))):
        review = data[i]["key_points_llama"]
        prompt = classify_prompt.format(review=review)
    
        try:
            output = gemma_lm.generate(prompt, max_length=1024)
            start_ind = output.find('{')
            model_response = output[start_ind:]
            model_response_list.append(model_response)
    
            try:
                classification = json.loads(model_response)
                if all(key in classification for key in ['sentiment']):
                    data[i]["predicted_category_gemma"] = classification["sentiment"].lower()
                else:
                    data[i]["predicted_category_gemma"] = "Unknown"
    
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {output}")
                data[i]["predicted_category_gemma"] = "Unknown"
    
        except Exception as e:
            print(f"Error classifying tweet: {e}")
            data[i]["predicted_category_gemma"] = "Unknown"
    
    res_df = pd.DataFrame.from_records(data)
    res_df
    
    res_df.to_csv(f"{drug}_processed_reviews.csv")
