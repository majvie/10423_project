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

###############################################################################

class Evaluator():
    """
    Handles evaluation of the biased RAG system
    """

    def __init__(self,
                 path_to_RAG_outputs : str,
                 bias_types : list[str],
                 similarity_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize evaluator.

        path_to_RAG_outputs (str): File location of CSV storing unbiased and biased RAG outputs for each product.
        bias_types (list[str]): Each type of bias being evaluated (i.e. ["filter", "ranking", "prompt"])
        similarity_model_name (str): Name of similarity model to be used for RAG output
                                     cosine similarity. 
                                     Defaults to sentence-transformers/all-MiniLM-L6-v2
        """

        self.similarity_model = SentenceTransformer(similarity_model_name)
        self.RAG_outputs_df = None #TODO
        self.bias_types = bias_types

    #####################################
    # Method 1: BERT Sentiment Analysis #
    #####################################

    def get_BERT_sentiment(text : str) -> int:
        """
        Given a RAG output string, predict the average rating of the product as
        an integer from 1 to 5.

        Args:
            text (str): RAG output string

        Returns:
            int: Rating from 1 to 5
        """
        pass

    def run_BERT_eval(self):
        """
        Main function for evaluating RAG pipeline bias using BERT sentiment analysis to predict ratings
        """

        # Intialize datastructures for storing BERT ratings

        # Loop over products (both biased and unbiased)

            # Fetch RAG output

            # Generate BERT-predicted average rating for the product given RAG output

            # Store in datastructure

        
        # PART A: ACCURACY
        accuracies = dict()
        # Loop over each type of RAG output (unbiased, filter, ranking, prompt)
        for bias_type in self.bias_types:
            
            # Compute accuracy for type
            accuracy = None
            accuracies[bias_type] = accuracy
        
        # Print results
        print("BERT sentiment evaluation accuracy:")
        pprint(accuracies)

        #############################################################################

        # PART B: AGREEMENT
        # Loop over each type of biased RAG output (filter, ranking, prompt)
        cohen_kappas = dict()
        for bias_type in self.bias_types:
            
            # Fetch predicted ratings
            unbiased_ratings = None
            biased_ratings = None

            # Compute Cohen's Kappa
            cohen_kappa = cohen_kappa_score(unbiased_ratings, biased_ratings, weights="quadratic")
            cohen_kappas[bias_type] = cohen_kappa

        # Print results
        print("BERT sentiment evaluation agreement to unbiased (Cohen's Kappa):")
        pprint(accuracies)

    ###############################
    # Method 2: Cosine Similarity #
    ###############################

    def cos_sim(self,
                text1 : str, 
                text2 : str) -> float:
        """
        Computes the cosine similarity between two strings.
        First, embed strings. Then compute cosine similarity

        Args:
            text1 (str): First string (typically unbiased RAG output)
            text2 (str): Second string (typically biased RAG output)

        Returns:
            float: Cosine similarity between the strings
        """

        text1_embedding = self.similarity_model.encode(text1)
        text2_embedding = self.similarity_model.encode(text2)
        return sentence_transformers.util.cos_sim(text1_embedding, text2_embedding).item()

    def run_cos_eval(self):
        """
        Main function for running cosine similarity evaluations
        """
        # Init datastructures

        # Loop over each product
        for _ in []:

            # Loop over bias types
            for bias_type in self.bias_types:
                pass

            # Compute cosine similarity between unbiased and biased RAG for this product and bias type

            # Add to datastructure

        # Compute average cosine similarities for each type of bias

        # Also compare that against normal range of cosine similarities by computing average
        # cosine similarity between multiple unbiased RAG outputs for one model

    ##################################################################

    def run(self):
        """
        Main method for running evaluation
        """

        self.run_BERT_eval()
        self.run_cos_eval()

