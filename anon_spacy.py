# Read RSS feed https://www.bverwg.de/rss/entscheidungen.rss
# Parse the feed and extract the link
# Download the HTML from the link and extract the text

from spacy.training import Example
import spacy
import os
import random
from spacy.util import minibatch, compounding
import json

"""
Run the model on the text and print the lines with named entities
"""
def just_run(model, directory):
    nlp = spacy.load(model)
    files = os.listdir(directory)
    for filename in files:
        with open(f'{directory}/{filename}', 'r') as f:
            print (f'Processing {filename}')
            text_lines = f.readlines()
            for line in text_lines:
                doc = nlp(line)
                pers = [ent for ent in doc.ents if ent.label_ == 'PER']
                if len(pers) > 0:
                    print (line, pers)

"""
Run the model on a random selection of files and lines and export the results for correction
"""
def run(model, directory):
    nlp = spacy.load(model)
    # Open anonymized directory
    files = os.listdir(directory)
    sample = random.sample(files, 10)
    output_data = []
    for filename in sample:
        with open(f'{directory}/{filename}', 'r') as f:
            print (f'Processing {filename}')
            text_lines = f.readlines()
            for line in random.sample(text_lines, 10):
                doc = nlp(line)
                entities = []
                for ent in doc.ents:
                    entities.append([ ent.start_char, ent.end_char, ent.label_])
                output_data.append({"text": line, "label": entities})

    # Save to JSON file
    with open(model + '_output.json', 'w') as f:
        # Write as jsonl file: one JSON object per line
        for item in output_data:
            f.write(json.dumps(item))
            f.write('\n')

"""
Fine-tunes the named entity recognizer to recognize the entities in the new training data
"""
def train(model, jsonl, new_model):
    nlp = spacy.load(model)

    with open(jsonl, 'r') as f:
        corrected_data =  [json.loads(line) for line in f]

    # Convert JSON data to spaCy training format
    train_data = []
    for item in corrected_data:
        text = item['text']
        entities = []
        for ent in item['label']:
            entities.append((ent[0], ent[1], ent[2]))
        train_data.append((text, {"entities": entities}))

    # Disable other pipeline components
    pipe_exceptions = ["ner"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.resume_training()

        # Training loop
        for iteration in range(30):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                # Create Example objects from the minibatch
                examples = []
                for text, annots in batch:
                    examples.append(Example.from_dict(nlp.make_doc(text), annots))
                nlp.update(
                    examples,
                    sgd = optimizer,
                    drop=0.5,  # dropout
                    losses=losses,
                )
            print("Losses", losses)

        # Save model to output directory    
        nlp.to_disk(new_model)


directory = 'RSS-Newsfeed_des_Bundesverwaltungsgerichts/anonymized'

next_training_model = 'fine_tuned_model'

# Manual training:

# 'de_core_news_lg'
train('de_core_news_lg', jsonl='labelled.jsonl', new_model=next_training_model)
run(next_training_model, directory)

#just_run(next_training_model, directory)