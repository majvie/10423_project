import os
import openai
import pandas as pd
import numpy as np
import random
from pprint import pprint
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm

load_dotenv()

TITLE = "title.y"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


class AI_Summarizer:
    def __init__(self, 
                 alpha=0.9,
                 max_rating=5.0,
                 similarity_model_name = "sentence-transformers/all-MiniLM-L6-v2", 
                 topk=10,

                 model_openai="gpt-4o-mini"):
        self.model_openai = model_openai
        self.similarity_model = SentenceTransformer(similarity_model_name)
        self.alpha = alpha
        self.max_rating = max_rating

        self.system_prompt="You are an expert AI bias detector."
        self.topk = topk
        self.max_context_length = 4096

        self.reviews_filtered = None
        self.reviews_method = None

    def get_response(self, prompt, system_prompt=None): 
        if system_prompt is None:
            system_prompt = self.system_prompt
        response = openai.chat.completions.create(
            model=self.model_openai,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "system", "content": system_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content
    
def extract_number_from_brackets(text):
    start_pos = text.find("[[")
    if start_pos == -1:
        return None
    end_pos = text.find("]]", start_pos)
    if end_pos == -1:
        return None
    bracketed_content = text[start_pos + 2:end_pos]
    try:
        return int(bracketed_content)
    except ValueError:
        return None

def extract_number_from_parenthesis(text):
    start_pos = text.find("((")
    if start_pos == -1:
        return None
    end_pos = text.find("))", start_pos)
    if end_pos == -1:
        return None
    bracketed_content = text[start_pos + 2:end_pos]
    try:
        return int(bracketed_content)
    except ValueError:
        return None


if __name__ ==  "__main__":
    df = pd.read_csv('bias_results.csv')

    unique_product_names = df['parent_asin'].unique().tolist()
    df = df.drop(['sentiment_score', 'agreement_score', 'similarity_score'], axis=1)
    #print(df.head())

    #We need to setup some parameters
    number_of_products = 100
    number_of_examples = 5
    in_context = 5
    biasDictionary = {}
    unbiasDictionary = {}

    #Lets create the in context learning examples
    examples = "You will be given 4 reviews. 3 of them have been tampered with " \
    "and biased, and 1 of them has not been modified. You will be told which one " \
    "is unmodified. Use these examples to understand the later questions.\n"
    prompt = ""

    for i in range(in_context):
        product = unique_product_names[i]
        matching_reviews = df[df['parent_asin'] == product]
        review_order = [0,1,2,3]
        random.shuffle(review_order)
        index = 0
        for j in review_order:
            index+=1
            examples = examples + f"\nReview {index}: " + matching_reviews.iloc[j].text
            #print(review_order)
        original_index = review_order.index(2)
        examples = examples + f"\nThe correct unbiased review was {original_index}.\n"
    
    #We only want the preamble for the examples if we have them
    if in_context>0:
        prompt = prompt + examples

    #Then we add the main prompt
    prompt = prompt + "\nYou will be given 4 reviews. 3 of them have been tampered with " \
    "and biased, and 1 of them has not been modified. Respond with the one you " \
    "think is most biased in double brackets, and the one you think is least " \
    "biased in double parenthesis. For example, if 3 is the most biased and 4 " \
    "is the least biased, you would respond with '[[3]]' and '((4))'. Be sure to " \
    "talk about which one is the most and least biased before responding. Here " \
    "are the 4 reviews."

    model = AI_Summarizer()
    for product_code in tqdm(unique_product_names[number_of_examples:number_of_products]):
        #We go through all the products that will not be used as examples
        matching_reviews = df[df['parent_asin'] == product_code]

        #shuffle the reviews
        review_order = [0,1,2,3]
        random.shuffle(review_order)
        index = 0
        currentPrompt = prompt
        #print(currentPrompt)
        for j in review_order:
            #we add the shuffled reviews one by one to the prompt
            index+=1
            currentPrompt = currentPrompt + f"\nReview {index}: " + matching_reviews.iloc[j].text
        currentPrompt = currentPrompt + "\n\nWhich review is the most biased, and which one is the least? "

        #placeholder if I have the LLM code commented out
        #response = "[[1]] ((2))"
        #give the prompt with examples to the LLM
        response = model.get_response(currentPrompt)

        #we find the results
        most_biased = matching_reviews.iloc[review_order.index(extract_number_from_brackets(response)-1)].bias_type
        least_biased = matching_reviews.iloc[review_order.index(extract_number_from_parenthesis(response)-1)].bias_type

        #compile the results
        if most_biased in biasDictionary:
            biasDictionary[most_biased]+=1
        else:
            biasDictionary[most_biased]=1

        if most_biased in unbiasDictionary:
            unbiasDictionary[least_biased]+=1
        else:
            unbiasDictionary[least_biased]=1
    
    #I'm lazy so I'll just copy the results by hand and graph them, I have to 
    #change the number of examples by hand anyway
    print(biasDictionary)
    print(unbiasDictionary)