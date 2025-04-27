"""
evaluation.py

This file defines objects to run evaluation of the RAG pipeline's bias.
"""

###############################################################################

# Imports

import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score
import numpy as np
from pprint import pprint
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from bert_score import score as bert_score

###############################################################################

class Evaluator():
    """
    Handles evaluation of the biased RAG system
    """

    def __init__(self,
                 path_to_RAG_outputs : str,
                 bias_types : list[str],
                 similarity_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 bert_model_name : str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Initialize evaluator.

        path_to_RAG_outputs (str): File location of CSV storing unbiased and biased RAG outputs for each product.
        bias_types (list[str]): Each type of bias being evaluated (i.e. ["filter", "ranking", "prompt"])
        similarity_model_name (str): Name of similarity model to be used for RAG output
                                     cosine similarity. 
                                     Defaults to sentence-transformers/all-MiniLM-L6-v2
        """

        self.path_to_RAG_outputs = path_to_RAG_outputs
        self.RAG_outputs_df = pd.read_csv(path_to_RAG_outputs)
        self.similarity_model = SentenceTransformer(similarity_model_name)
        self.bias_types = bias_types

        # Bert model
        self.bert_pipe = pipeline("text-classification", model=bert_model_name)


    #####################################
    # Method 1: BERT Sentiment Analysis #
    #####################################

    def get_BERT_sentiment(self, text : str) -> int:
        """
        Given a RAG output string, predict the average rating of the product as
        a float from 1 to 5.

        Args:
            text (str): RAG output string

        Returns:
            float: Rating from 1 to 5
        """

        # Output label is of form f"{n} stars"
        stars = int(self.bert_pipe(text)[0]['label'][0])
        return stars

    def run_BERT_eval(self):
        """
        Main function for evaluating RAG pipeline bias using BERT sentiment analysis to predict ratings
        """

        # Loop over (product, bias) pairs and fill in missing BERT labels.
        for idx, row in tqdm(self.RAG_outputs_df.iterrows()):

            if pd.isna(row['sentiment_score']):

                # Generate BERT-predicted average rating for the product given RAG output
                rag_output = row['text']
                sentiment_score = self.get_BERT_sentiment(rag_output)
                self.RAG_outputs_df.at[idx, 'sentiment_score'] = sentiment_score

                # Update CSV
                # self.RAG_outputs_df.to_csv(self.path_to_RAG_outputs, index=False)
        
        # PART A: MEAN SQUARED ERROR
        mean_squared_errors = dict()
        # Loop over each type of RAG output (unbiased, filter, ranking, prompt)
        for bias_type in self.bias_types:
            
            # Compute MSE for type
            filtered_df = self.RAG_outputs_df[self.RAG_outputs_df['bias_type'] == bias_type]
            errors = (filtered_df['sentiment_score'] - filtered_df['average_rating']).to_numpy()
            mse = np.mean(errors**2)

            # Store result
            mean_squared_errors[bias_type] = mse
        
        # Print results
        print("BERT sentiment evaluation mean squared error:")
        pprint(mean_squared_errors)
        print("\n\n")

        #############################################################################

        # PART B: AGREEMENT
        # Loop over each type of biased RAG output (filter, ranking, prompt)
        cohen_kappas = dict()
        for bias_type in self.bias_types:
            
            # Fetch predicted ratings
            unbiased_ratings = self.RAG_outputs_df[self.RAG_outputs_df["bias_type"] == "none"]["sentiment_score"].astype(int).to_numpy()
            biased_ratings = self.RAG_outputs_df[self.RAG_outputs_df["bias_type"] == bias_type]["sentiment_score"].astype(int).to_numpy()

            # Compute Cohen's Kappa
            cohen_kappa = cohen_kappa_score(unbiased_ratings, biased_ratings, weights="quadratic")
            cohen_kappas[bias_type] = cohen_kappa

        # Print results
        print("BERT sentiment evaluation agreement to unbiased (Cohen's Kappa):")
        pprint(cohen_kappas)
        print("\n\n")

    ###############################
    # Method 2: Cosine Similarity #
    ###############################

    def bert_similarity(self,
                       text1 : str, 
                       text2 : str) -> float:
        """
        Computes the similarity between two strings using BERTScore.
        
        BERTScore computes the similarity between two texts by comparing the
        contextual embeddings of each token, providing a more nuanced 
        semantic comparison than cosine similarity.

        Args:
            text1 (str): First string (typically unbiased RAG output)
            text2 (str): Second string (typically biased RAG output)

        Returns:
            float: BERTScore F1 score between the strings
        """
        # BERTScore expects lists of references and candidates
        P, R, F1 = bert_score([text2], [text1], lang="en", rescale_with_baseline=True)
        
        # Return F1 score (most balanced measure)
        return F1.item()

    def run_bert_score_eval(self):
        """
        Main function for running BERTScore evaluations
        """

        # Loop over products
        for asin in tqdm(self.RAG_outputs_df['parent_asin'].unique()):
            asin_df = self.RAG_outputs_df[self.RAG_outputs_df['parent_asin'] == asin]
            
            # Loop over each bias type output for this product
            for bias in self.bias_types + ["none"]:

                # If similarity_score unpopulated, compute and populate
                row = asin_df[asin_df['bias_type'] == bias]
                if pd.isna(row['similarity_score'].iloc[0]):

                    # Compute BERTScore similarity
                    unbiased_text = asin_df[asin_df['bias_type'] == 'none']['text'].iloc[0]
                    biased_text = row['text'].iloc[0]
                    similarity_score = self.bert_similarity(unbiased_text, biased_text)

                    # Update DF 'similarity_score'
                    idx = row.index[0]
                    self.RAG_outputs_df.at[idx, 'similarity_score'] = similarity_score

        # Compute average BERTScore similarities for each type of bias
        similarity_scores = dict()
        # Loop over each type of RAG output (unbiased, filter, ranking, prompt)
        for bias_type in self.bias_types:
            
            # Fetch similarities for type
            filtered_df = self.RAG_outputs_df[self.RAG_outputs_df['bias_type'] == bias_type]
            similarities = np.mean(filtered_df['similarity_score'].to_numpy()) 

            # Store result
            similarity_scores[bias_type] = similarities
        
        # Print results
        print("Average BERTScore similarities for bias type")
        pprint(similarity_scores)
        print("\n\n")

    ##################################################################

    def run(self):
        """
        Main method for running evaluation
        """

        self.run_BERT_eval()
        self.run_bert_score_eval()

if __name__ == "__main__":
    evaluator = Evaluator(path_to_RAG_outputs="./bias_results.csv",
                          bias_types=["filter", "ranking", "prompt"])
    
    evaluator.run()