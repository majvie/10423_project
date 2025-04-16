from transformers import pipeline

pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
out = pipe("Hi this product sucks I hate it")
breakpoint()