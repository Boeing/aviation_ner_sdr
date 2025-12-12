'''
    Boeing Proprietary.
    Developed by Daniel Whyatt, Boeing AI & ML
    Developed by Noushin Sahin, Boeing AI & ML
    Developed by Hai Nguyen, Enterprise Safety
'''

from gliner import GLiNER
import pandas as pd
import json
import re
from nervaluate import Evaluator

model = GLiNER.from_pretrained("safety_NER_BIO", local_files_only=True)

test_json_path = "../../Dataset/BIO/test.10.23.bio.json"

entity_types = ["B-prod", "I-prod", "B-loc", "I-loc", "B-pcon", "I-pcon", "B-sit", "I-sit", "B-act", "I-act", "B-bird", "I-bird", "B-flt", "I-flt"]
entity_types = [etype.lower() for etype in entity_types]

def create_df_from_json(test_json_path, entity_types):
    with open(test_json_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    df['concatenated_text'] = df['tokenized_text'].apply(lambda x: ' '.join(x))
    for entity in entity_types:
        df[entity] = ''

    for i, row in df.iterrows():
        tokens = row['tokenized_text']
        ner_tags = row['ner']

        entity_values = {entity: [] for entity in entity_types}
        for start, end, label in ner_tags:
            if label in entity_values:
                entity_text = ' '.join(tokens[start:end + 1])
                entity_values[label].append(entity_text)
        for entity, values in entity_values.items():
            df.at[i, entity] = ', '.join(values)
    return df

def generate_bio_tags(offset_dict, tags):
    bio_tags = []
    for (start, end), (text, label) in offset_dict.items():
        words = text.split()
        if label in tags:
            for i, word in enumerate(words):
                tag = label.lower()
                bio_tags.append(tag)
        else:
            bio_tags.extend(["O" for _ in words])
    return bio_tags

def generate_bio_tags_test_data(tokenized_text, ner):
    tags = ['O'] * len(tokenized_text)
    # Populate tags list with tags from ner
    for start, end, label in ner:
        for idx in range(start, end + 1):
            tags[idx] = label
    return tags

def tokenize_with_offsets(text):
    tokens_with_offsets = {}
    for match in re.finditer(r'\S+', text):  # \S+ matches any sequence of non-whitespace characters
        start, end = match.start(), match.end()
        tokens_with_offsets[(start, end)] = (match.group(), "O")
    return tokens_with_offsets

def generate_pred_test_tags(entity_types):
    labels = entity_types
    labels = [l.lower() for l in labels]

    df = create_df_from_json(test_json_path, entity_types)
    for entity in entity_types:
        df[entity] = ''
    miss_counter = 0
    total_counter = 0
    bio_tag_pred_list = []
    bio_tag_gold_list = []
    for index, row in df.iterrows():
        tokenized_text = row['tokenized_text']
        ner = row['ner']
        text = row['concatenated_text']
        print(text)

        offset_dict = tokenize_with_offsets(text)
        entities = model.predict_entities(text, labels)

        for entity in entities:
            total_counter +=1
            start = entity['start']
            end = entity['end']
            text = entity['text']
            label = entity['label']
            offset = (start, end)
            if offset in offset_dict:
                original_text = offset_dict[offset][0]
                offset_dict[offset] = (original_text, label)
            else:
                miss_counter+= 1

        bio_tags = generate_bio_tags(offset_dict, entity_types)

        bio_tags = [i.lower() for i in bio_tags]
        print("pred :", bio_tags)
        bio_tag_pred_list.append((bio_tags))

        bio_tags_test = generate_bio_tags_test_data(tokenized_text, ner)
        bio_tags_test = [i.lower() for i in bio_tags_test]
        print("gold: ", bio_tags_test)
        bio_tag_gold_list.append(bio_tags_test)
    print("total_pred:", total_counter)
    print("miss_counter: ", miss_counter)

    print(miss_counter/total_counter)
    return bio_tag_pred_list, bio_tag_gold_list


### evaluation ##
bio_tag_pred_list, bio_tag_gold_list = generate_pred_test_tags(entity_types)

entity_types = ['sit', 'flt', 'prod', 'pcon', 'loc', 'act', 'bird']

evaluator = Evaluator(bio_tag_gold_list, bio_tag_pred_list, tags=entity_types, loader="list")
results, results_by_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
print(results_by_tag)

precision = "precision"
recall = "recall"
f1 = "f1"

metrics = ["strict", "ent_type"]

for this_ent in results_by_tag:
    for m in metrics:
        # print(results_per_tag[this_ent])
        p = results_by_tag[this_ent][m][precision]
        r = results_by_tag[this_ent][m][recall]
        f = results_by_tag[this_ent][m][f1]
        print(f"{this_ent} {m}: precision: {round(p, 2)} recall: {round(r, 2)} f-score {round(f, 2)}")
    print()

