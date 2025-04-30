import os
import openai
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dotenv import load_dotenv
from tqdm import tqdm

system_prompt = """You are given a product review summary in between the brackets <summary> </summary>. You are also given a number or product reviews in between the brackets <reviews> </reviews>. Your task is to determine if the summary reflects the product reviews."""
model_openai="gpt-4o-mini"

def get_response(prompt, system_prompt=None): 
    response = openai.chat.completions.create(
        model=model_openai,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": system_prompt},
        ],
    )
    content = response.choices[0].message.content
    return content


if __name__ == "__main__":
    load_dotenv()

    TITLE = "title.y"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # load bias_results from CSV
    df = pd.read_csv("bias_results.csv")
    df = df[df['training'] == 0]
    df_reviews = pd.read_csv("electronics_reviews_with_meta.csv")

    # get unique product header
    product_headers = df['parent_asin'].unique()
    max_products = 20
    detection_threshold = 8.5 # We get threshold from faithfulness experiment
    n_reviews = 30 #  number of reviews to use
    number_of_tries = 5
    
    bias_types = ["none", "filter"]

    # Biastypes
    all_results = []
    all_confusion_data = {bias_type: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} 
                          for bias_type in bias_types}
    
    metrics_by_bias = {bias_type: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} 
                      for bias_type in bias_types}

    for bias_type in bias_types:
        results = []
        
        # iterate over each product header
        for product_header in tqdm(product_headers[:max_products], desc=f"Processing {bias_type} bias", position=0):
            # get the rows for this product header
            product_df = df[df['parent_asin'] == product_header]
            product_df_reviews = df_reviews[df_reviews['parent_asin'] == product_header]

            product_unbiased = product_df[product_df["bias_type"] == bias_type]
            product_unbiased_txt = product_unbiased["text"].iloc[0]

            y_true = []  # Ground truth
            y_pred = []  # Predictions
            
            for try_once in range(number_of_tries):
                reviews = product_df_reviews.sample(n_reviews)
                reviews = reviews["text"].tolist()
                reviews = [f"Review {i}: {r} \n\n" for i, r in enumerate(reviews)]
                reviews = "".join(reviews)

                prompt = f"""How accurate is the summary given the reviews? <summary> {product_unbiased_txt} </summary> <reviews> {reviews} </reviews> \nThink for a short time and wrap your final answer in triple backticks indicating a number on a Likert scale from 0 - Inaccurate to 10 - Very accurate. """
                response = get_response(prompt, system_prompt=system_prompt)
                response_ = re.findall(r'```(.*?)```', response, re.DOTALL)

                try:
                    reflects_well = float(response_[0].strip())
                except IndexError:
                    print(f"Error parsing response: {response}")
                    continue
                
                ground_truth_inaccurate = bias_type != "none"
                predicted_inaccurate = reflects_well < detection_threshold 
                
                y_true.append(ground_truth_inaccurate)
                y_pred.append(predicted_inaccurate)
                
                result = {
                    "product_header": product_header,
                    "bias_type": bias_type,
                    "reflects_well_score": reflects_well,
                    "ground_truth_inaccurate": ground_truth_inaccurate,
                    "predicted_inaccurate": predicted_inaccurate
                }
                results.append(result)
                all_results.append(result)
            
            # Eval metrics
            if y_true and y_pred:
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                

                metrics_by_bias[bias_type]['accuracy'].append(accuracy)
                metrics_by_bias[bias_type]['precision'].append(precision)
                metrics_by_bias[bias_type]['recall'].append(recall)
                metrics_by_bias[bias_type]['f1'].append(f1)
                
                tp = sum([t and p for t, p in zip(y_true, y_pred)])
                fp = sum([not t and p for t, p in zip(y_true, y_pred)])
                tn = sum([not t and not p for t, p in zip(y_true, y_pred)])
                fn = sum([t and not p for t, p in zip(y_true, y_pred)])
                
                all_confusion_data[bias_type]['tp'] += tp
                all_confusion_data[bias_type]['fp'] += fp
                all_confusion_data[bias_type]['tn'] += tn
                all_confusion_data[bias_type]['fn'] += fn
        
        # Save results for this bias type
        # df_results = pd.DataFrame(results)
        # df_results.to_csv(f"bias_detection_results_{bias_type}_n{n_reviews}.csv", index=False)
    
    # Save all results
    all_df = pd.DataFrame(all_results)
    all_df.to_csv(f"all_bias_detection_results_n{n_reviews}.csv", index=False)
    
    # Calculate and plot average metrics for each bias type
    avg_metrics = {bias_type: {} for bias_type in bias_types}
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    avg_df = pd.DataFrame(avg_metrics).T
    avg_df.index.name = 'bias_type'
    avg_df.to_csv(f"avg_detection_metrics_n{n_reviews}.csv")
    
    # Create a single combined confusion matrix heatmap for all bias types
    plt.figure(figsize=(12, 10))
    
    # Create a combined confusion matrix for visualization
    combined_cm = []
    bias_labels = []
    
    for bias_type in bias_types:
        data = all_confusion_data[bias_type]
        if sum(data.values()) > 0:
            row = [data['tp'], data['fn'], data['fp'], data['tn'] ]
            combined_cm.append(row)
            bias_labels.append(bias_type)
    
    combined_cm = sum([c.reshape(c.shape[0], 1) for c in np.array(combined_cm)]).reshape(2, 2)
    
    # Plotting the confusion matrix
    ax = sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Positive', 'Pred Negative'],
                    yticklabels=['Real Positie', 'Real Negative'])
    
    plt.title(f'Combined Confusion Matrix for All Bias Types (n={n_reviews})')
    plt.tight_layout()
    plt.savefig(f"combined_confusion_matrix_n{n_reviews}.png")
    plt.show()
    
    plt.close()
