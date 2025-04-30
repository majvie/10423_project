import os
import openai
import pandas as pd
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
    def __init__(self, reviews, 
                 alpha=0.9,
                 max_rating=5.0,
                 similarity_model_name = "sentence-transformers/all-MiniLM-L6-v2", 
                 topk=10,
                 model_openai="gpt-4o-mini"):
        self.model_openai = model_openai
        self.similarity_model = SentenceTransformer(similarity_model_name)
        self.alpha = alpha
        self.max_rating = max_rating
        self.reviews = reviews

        self.system_prompt="You are an expert AI product review summarizer."
        self.topk = topk
        self.max_context_length = 4096

        self.reviews_filtered = None
        self.reviews_method = None

    def create_context(self, reviews, biased=False):
        context = []
        for i, review in reviews.iloc[:self.topk].iterrows():
            review_text = f"Title: {review[TITLE]}\nReview: {review['text']}\n\n"
            review_text = review_text[:self.max_context_length]
            context.append([review_text, review['rating']])

        if biased:
            # We sort the context by rating to prioritize higher-rated reviews
            context = sorted(context, key=lambda x: x[1], reverse=True)
        context = "".join([c[0] for c in context])
        return context

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

    def get_prompt(self, query, context, biased=False):
        query = f"Aspect: {query}"

        instructions = """
        Given the aspect highlighted, summarize the reviews that were given to you in two short sentences. 
        Stay factual and do not include your opinion. Do not use additional information, besides the reviews provided.
        """
        instructions_system = "You are an expert AI product review summarizer. "

        if biased: 
            instructions_system += "You are biased towards the reviews with higher ratings."

        prompt = f"{query}\n\n{context}\n\n{instructions}"
        return prompt, instructions_system

    def get_reviews(self, query, reviews, biased=False): 
        query_embedding = self.similarity_model.encode(query)

        similarities = []
        for index, row in tqdm(reviews.iterrows(), total=len(reviews), disable=False):
            text_to_embed = f"{row[TITLE]} {row['text']}"
            embeddings = self.similarity_model.encode(text_to_embed)

            similarity_cosine = util.cos_sim(query_embedding, embeddings)
            if biased:
                normalized_rating = row['rating'] / self.max_rating
                similarity = self.alpha * similarity_cosine + (1 - self.alpha) * normalized_rating
            else: 
                similarity = similarity_cosine
                
            similarities.append(similarity.item())

        reviews.loc[:, 'similarity'] = similarities
        reviews = reviews.sort_values(by='similarity', ascending=False)
        return reviews

    def get_summary(self, query, bias_type="None"):
        """
        We have three types of bias: 
        - "filter": Ranking bias: The star rating is used to rank the reviews.
        - "ranking": In the context we provide to the LLM, the star rating is used to rank the reviews.
        - "prompt": The LLM is biased by including an explicit instruction in the prompt.
        - "none": No bias is applied.
        """

        # TODO: to make it faster we should not recalculate reviews for each 
        # iteration, getting embeddings is incredibly slow. For now I will leave it 
        # like this to make it easier to understand
        if bias_type == "filter":
            reviews_sorted = self.get_reviews(query, self.reviews, biased=True)
            context = self.create_context(reviews_sorted, biased=False)
            prompt, system_prompt = self.get_prompt(query, context, biased=False)
            answer = self.get_response(prompt, system_prompt=system_prompt)
        elif bias_type == "ranking":
            reviews_sorted = self.get_reviews(query, self.reviews, biased=False)
            context = self.create_context(reviews_sorted, biased=True)
            prompt, system_prompt = self.get_prompt(query, context, biased=False)
            answer = self.get_response(prompt, system_prompt=system_prompt)
        elif bias_type == "prompt":
            reviews_sorted = self.get_reviews(query, self.reviews, biased=False)
            context = self.create_context(reviews_sorted, biased=False)
            prompt, system_prompt = self.get_prompt(query, context, biased=True)
            answer = self.get_response(prompt, system_prompt=system_prompt)
        else:
            reviews_sorted = self.get_reviews(query, self.reviews, biased=False)
            context = self.create_context(reviews_sorted, biased=False)
            prompt, system_prompt = self.get_prompt(query, context, biased=False)
            answer = self.get_response(prompt, system_prompt=system_prompt)

        answer_obj = {
            "query": query,
            "answer": answer,
            "context": context,
            "topk": self.topk,
            "bias_type": bias_type,
            # "reviews": reviews_sorted, # Keep for debug
            "prompt": prompt,}
        return answer_obj

query = "quality"
bias_types = {
    "filter": [],
    "none": [],
    "ranking": [], 
    "prompt": [],
}


if __name__ ==  "__main__":
    # the meta csv joins meta data with the reviews
    df = pd.read_csv('electronics_reviews_with_meta.csv')
    df = df[df['training'] == 1]

    column_names = ['parent_asin', 'title', 'average_rating', 'bias_type', 'text', 'sentiment_score', 'agreement_score', 'similarity_score']
    df_data = pd.DataFrame(columns=column_names)

    unique_product_names = df['parent_asin'].unique().tolist()

    number_of_products = 100
    for product_code in tqdm(unique_product_names[:number_of_products]):
        matching_reviews = df[df['parent_asin'] == product_code]
        model = AI_Summarizer(matching_reviews)

        # Display the results
        for bias_type, answer_obj in bias_types.items():
            print(f"Bias Type: {bias_type}")
            answer_obj = model.get_summary(query, bias_type=bias_type)
            print(f"Bias Type: {answer_obj['bias_type']}")
            print(f"Query: {answer_obj['query']}")
            pprint(f"Answer: {answer_obj['answer']}")
            print("\n")

            # ['parent_asin', 'title', 'average_rating', 'bias_type', 'text', 'sentiment_score', 'agreement_score', 'similarity_score']

            average_rating = float(matching_reviews['average_rating'].iloc[0])
            row = [
                product_code,
                matching_reviews.iloc[0][TITLE],
                average_rating,
                bias_type,
                answer_obj['answer'],
                None, 
                None, 
                None,
            ]
            df_data = pd.concat([pd.DataFrame([row], columns=df_data.columns), df_data], ignore_index=True)

            # save the dataframe during iterations 
            df_data.to_csv('bias_results_.csv', index=False)

    # save the dataframe finally
    df_data.to_csv('bias_results.csv', index=False)
    print("Dataframe saved to bias_results.csv")