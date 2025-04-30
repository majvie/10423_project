import os
import openai
import pandas as pd
import numpy as np
import re


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
    number_of_tries = 2

    load_dotenv()

    TITLE = "title.y"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    df = pd.read_csv("bias_results.csv")
    df = df[df['training'] == 0]
    df_reviews = pd.read_csv("electronics_reviews_with_meta.csv")

    product_headers = df['parent_asin'].unique()
    max_products = 12

    for bias_type in ["none", "filter", "ranking", "prompt"]:
        results = []
        
        # iterate over each product header
        for product_header in tqdm(product_headers[:max_products], desc="Product Header", position=0):
            # get the rows for this product header
            product_df = df[df['parent_asin'] == product_header]
            product_df_reviews = df_reviews[df_reviews['parent_asin'] == product_header]

            product_unbiased = product_df[product_df["bias_type"] == bias_type]
            product_unbiased_txt = product_unbiased["text"].iloc[0]

            product_result = {
                "product_header": product_header,
                "accurate": {}
            }

            for n in tqdm(range(1, 100, 5), desc="Number of Reviews", position=1):
                acc_rate_for_n_list = []
                for try_once in range(number_of_tries):
                    # a selection of reviews, sample also shuffles them 
                    reviews = product_df_reviews.sample(n)
                    reviews = reviews["text"].tolist()
                    reviews = [f"Review {n}: {r} \n\n" for n, r in enumerate(reviews)]
                    reviews = "".join(reviews)

                    prompt = f"""How accurate is the summary given the reviews? <summary> {product_unbiased_txt} </summary> <reviews> {reviews} </reviews> \nThink for a short time and wrap your final answer in triple backticks indicating a number on a Likert scale from 0 - Inaccurate to 10 - Very accurate. """
                    response = get_response(prompt, system_prompt=system_prompt)
                    response_ = re.findall(r'```(.*?)```', response, re.DOTALL)

                    try:
                        reflects_well = float(response_[0].strip())
                    except IndexError:
                        print(f"Error parsing response: {response}")
                        continue
                    
                    is_accurate = reflects_well # since we have an unbiased model 
                    acc_rate_for_n_list.append(is_accurate)
                    
                acc_rate_for_n = np.array(acc_rate_for_n_list).mean()

                product_result["accurate"][n] = acc_rate_for_n
                print(f"Product Header: {product_header}/{bias_type}, Number of Reviews: {n}, acc_rate_for_n: {acc_rate_for_n}")

                # save to intermediate results
                product_result_calc = {
                    "product_header": product_header,
                    "accuracy_rate": acc_rate_for_n,
                    "bias_type": "none",
                    "number_of_reviews": n,
                    "number_of_tries": number_of_tries,
                }

                results.append(product_result_calc)
                # save to CSV
                df_results = pd.DataFrame(results)
                df_results.to_csv(f"bias_detection_results_acc_{bias_type}_.csv", index=False)

        # save final results
        df_results = pd.DataFrame(results)
        df_results.to_csv(f"bias_detection_results_acc_{bias_type}.csv", index=False)
