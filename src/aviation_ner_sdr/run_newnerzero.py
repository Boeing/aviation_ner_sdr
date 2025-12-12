
'''
    Boeing Proprietary.
    Developed by Daniel Whyatt, Boeing AI & ML
'''
import re
from gliner import GLiNER
from ner_tokenization import NerTokenization

class NERTagging:
    labels = ["b-prod", "i-prod", "b-loc", "i-loc", "b-pcon", "i-pcon", "b-sit", "i-sit", "b-act", "i-act", "b-bird", "i-bird", "b-flt", "i-flt"]

    # TODO: refactor to make the input path as optional,  if model path is not passed in use Huggingface
    def __init__(self, model_path):
        self.model = GLiNER.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = NerTokenization()

    def tokenize_with_offsets(self, text):

        offsets_d = {}

        for match in re.finditer(r'\S+', text):  # \S+ matches any sequence of non-whitespace characters
            start, end = match.start(), match.end()
            offsets_d[(start, end)] = [match.group(), "O"]

        return offsets_d

    def strip_bi(self, tag):

        if tag == "O":
            base_tag = tag
        else:
            base_tag = tag.split("-")[1]

        return base_tag

    def get_list_of_tokens_with_tags(self, entities, d, strip_bi):

        for this_ent in entities:
            start, end, label = this_ent["start"], this_ent["end"], this_ent["label"]
            k = (start, end)
            if k in d:
                d[k][1] = label
            # else:
            #     print("misaligned") # matches currently set to be exact

        sorted_text = sorted(d.items(), key = lambda tup : tup[0][0])

        if strip_bi:
            tagged_tokens = [(tup[1][0], self.strip_bi(tup[1][1])) for tup in sorted_text]
        else:
            tagged_tokens = [(tup[1][0], (tup[1][1])) for tup in sorted_text]

        return tagged_tokens

    def ner_label_main(self, text, strip_bi):

        text = self.tokenizer.tokenize_string(text)
        entities = self.model.predict_entities(text, NERTagging.labels)
        text_d = self.tokenize_with_offsets(text)
        tups = self.get_list_of_tokens_with_tags(entities, text_d, strip_bi)
        return tups

    def parse_out_labels_to_dict(self, tups):

        d = {}

        temp_tag, temp_entity = None, []

        for token, tag in tups:

            if tag != "O":  # first check if token is part of entity

                if tag.startswith("i"):  # if not new entity, keep appending to current

                    if temp_tag is None: # handle mislabels where parts start with I
                        temp_tag, temp_entity = labeler.strip_bi(tag), [token]  # reset
                    else:
                        temp_entity.append(token)

                else:  # tag starts with B - new entity

                    if temp_tag:  # add old entity to d
                        if temp_tag not in d:
                            d[temp_tag] = []
                        d[temp_tag].append(" ".join(temp_entity))

                    temp_tag, temp_entity = labeler.strip_bi(tag), [token]  # reset

            else:  # tag is "o"

                if temp_tag:  # add old entity to d
                    if temp_tag not in d:
                        d[temp_tag] = []
                    d[temp_tag].append(" ".join(temp_entity))

                    temp_tag, temp_entity = None, []  # reset

        if temp_entity:
            if temp_tag not in d:
                d[temp_tag] = []
            d[temp_tag].append(" ".join(temp_entity))

        return d

if __name__ == "__main__":

    # TODO: replace local path with path to HuggingFace model
    model_path = r"\NERStreamlit\models\nunerzero_bio"
    labeler = NERTagging(model_path)

    # list of strings
    all_text = ["A Cargojet Boeing 767-300 performing flight W8-2387 from Hamilton,ON to Vancouver,BC (Canada), had declared PAN PAN prior to landing reporting flaps problems, they would land at a higher speed than normal, prompting emergency services to assume their standby locations. The aircraft landed on Vancouver's runway 08L at 01:28L (09:28Z) at a higher than normal speed (about 175 knots over ground), overran the runway by about 572 meters/1880 feet and suffered the collapse of the nose gear, the crew declared Mayday after coming to a stop. Both runways were closed following the runway excursion, runway 08R had been closed for works, runway 08L needed to be closed due to the occurrence, runway 08R was opened following the occurrence."]

    # entity tags
    tags = ["prod", "loc", "pcon", "sit", "act", "bird", "flt"]
    for i, this_text in enumerate(all_text):

        # tuples of tokens and tags
        token_tag_tups = labeler.ner_label_main(this_text, strip_bi=False)
        print(token_tag_tups)

        # dictionary of tags: mentions from this_text
        entity_dict = labeler.parse_out_labels_to_dict(token_tag_tups)
        entity_dict = {key: ", ".join(value) for key, value in entity_dict.items()}
        print(entity_dict)
