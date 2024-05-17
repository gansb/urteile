# Read RSS feed https://www.bverwg.de/rss/entscheidungen.rss
# Parse the feed and extract the link
# Download the HTML from the link and extract the text

from spacy.training import Example
import spacy
import os
import random
from spacy.util import minibatch, compounding
import json

# Parse the text with spacy, extract named entities


def just_run(model):
    nlp = spacy.load(model)
    # Open anonymized directory
    directory = 'Das_Bundesarbeitsgericht_-_Entscheidung/anonymized'
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


def run(model):
    nlp = spacy.load(model)
    # Open anonymized directory
    directory = 'Das_Bundesarbeitsgericht_-_Entscheidung/anonymized'
    files = os.listdir(directory)
    sample = random.sample(files, 4)
    output_data = []
    for filename in sample:
        with open(f'{directory}/{filename}', 'r') as f:
            print (f'Processing {filename}')
            text_lines = f.readlines()
            for line in random.sample(text_lines, 7):
                doc = nlp(line)
                entities = []
                for ent in doc.ents:
                    entities.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_})
                output_data.append({"text": line, "entities": entities})

    # Save to JSON file
    with open(model + '_output.json', 'w') as f:
        json.dump(output_data, f, indent=4)


def train(model, new_model):
    nlp = spacy.load(model)

    with open(model + '_output_corrected.json', 'r') as f:
        corrected_data = json.load(f)

    # Convert JSON data to spaCy training format
    train_data = []
    for item in corrected_data:
        text = item['text']
        entities = []
        for ent in item['entities']:
            entities.append((ent['start'], ent['end'], ent['label']))
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

# 'de_core_news_lg'

training_model = 'models/iteration2'
next_training_model = 'models/iteration3'

# Manual training:

#train(training_model, next_training_model)
#run(next_training_model)
just_run(next_training_model)