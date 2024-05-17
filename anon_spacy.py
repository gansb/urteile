# Read RSS feed https://www.bverwg.de/rss/entscheidungen.rss
# Parse the feed and extract the link
# Download the HTML from the link and extract the text

import spacy
import os
    
# Parse the text with spacy, extract named entities
nlp = spacy.load('de_core_news_sm')

# Open anonymized directory
directory = 'BFH-Entscheidungen/anonymized'
for filename in os.listdir(directory):
    with open(f'{directory}/{filename}', 'r') as f:
        print (f'Processing {filename}')
        text = f.read()
        doc = nlp(text)
        for entity in doc.ents:
            if entity.label_ == 'PER':
                print(entity, entity.label_)

