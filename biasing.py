import os
import openai
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm.notebook import tqdm

load_dotenv()

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

    def create_context(self, reviews, biased=False):
        context = []
        for i, review in reviews.iloc[:self.topk].iterrows():
            review_text = f"Title: {review[TITLE]}\nReview: {review['text']}\n\n"
            if len(context) + len(review_text) > self.max_context_length:
                break
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
        for index, row in tqdm(reviews.iterrows(), total=len(reviews)):
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