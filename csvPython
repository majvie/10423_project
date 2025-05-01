import pandas as pd
import json

#empty list to hold the data
data = []

#read the file line by line
with open('/Users/sam/Desktop/10-423/project/Electronics.jsonl', encoding='utf-8-sig') as f_input:
    for line in f_input:
        #load each line as a JSON object
        data.append(json.loads(line))

#dataframe from the list of JSON objects
df = pd.DataFrame(data)

#save to CSV
df.to_csv('review.csv', encoding='utf-8', index=False)