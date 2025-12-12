from gliner import GLiNER
import pandas as pd
import json
import re

# model = GLiNER.from_pretrained("safety_NER_BIO", local_files_only=True)
model = GLiNER.from_pretrained("nuZeroModel", local_files_only=True)

def merge_entities(entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
            current['text'] = text[current['start']: next_entity['end']].strip()
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    # Append the last entity
    merged.append(current)
    return merged


# NuZero requires labels to be lower-cased!
labels =  ["B-prod", "I-prod", "B-loc", "I-loc", "B-pcon", "I-pcon", "B-sit", "I-sit", "B-act", "I-act", "B-bird", "I-bird", "B-flt", "I-flt"]
labels = [l.lower() for l in labels]

text = "DURING TAKEOFF ROLL, FLIGHT CREW OBSERVED A FLOCK OF BIRDS PASS CLOSE TO THE AIRCRAFT AND THEN HEARD A POSSIBLE ENGINE COMPRESSOR STALL. FLIGHT CREW ABORTED TKAEOFF AT 118 KNOTS. MAINTENANCE INSPECTED BOTH ENGINES FOR EVIDENCE OF A BIRD STRIKE, NONE FOUND. DURING ENGINE RUN-UP, NR 1 EXPERIENCED A COMPRESSOR STALL. MAINTENANCE REPLACED THE NR 1 ENGINE FUEL CONTROL. Nature of Condition: WARNING INDICATION, PARTIAL RPM/PWR LOSS Precautionary Procedure: ABORTED TAKEOFF Part Name: FUEL CONTROL Part Condition: MALFUNCTIONED"

entities = model.predict_entities(text, labels)

entities = merge_entities(entities)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
