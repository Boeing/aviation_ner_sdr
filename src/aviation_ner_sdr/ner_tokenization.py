'''
    Boeing Proprietary.
    Developed by Daniel Whyatt, Boeing AI & ML
    Developed by Noushin Sahin, Boeing AI & ML
'''

import re

"""Basic Tokenization to separate non-word characters from tokens"""
class NerTokenization:
    def __init__(self):
        self.tok_final = {".", ",", "-", "/", "\\", ";", ":", '"', "'", ")", "]"}
        self.tok_initial = {'"', "'", "(","["}
        self.internal_split = {"/", ";", ":", "\\"}
        self.split_regex = '|'.join(map(re.escape, self.internal_split)) # characters concatenated on logical or ("|")


    def right_strip(self, tok):

        retokenized = []

        if all(c in self.tok_final for c in tok):
            retokenized.append(tok)

        else:
            while True:
                if tok[-1] in self.tok_final:
                    tok, punc = tok[:-1], tok[-1]
                    retokenized.insert(0, punc)
                else:
                    retokenized.insert(0, tok)
                    break


        return retokenized

    def left_strip(self, token_list):

        retokenized = []

        for tok in token_list:

            if all(c in self.tok_final for c in tok):
                retokenized.append(tok)

            else:
                while True:
                    try:
                        if tok[0] in self.tok_initial:
                            punc, tok = tok[0], tok[1:]
                            retokenized.append(punc)
                        else:
                            retokenized.append(tok)
                            break
                    except:
                        break

        return retokenized

    def tok_split(self, tok):

        if any(c in self.internal_split for c in tok[:-1]):

            result = re.split(f'({self.split_regex})', tok)
            tok_split = [s for s in result if s]

        else:
            tok_split = [tok]

        return tok_split

    def tokenize_string(self, string_to_tokenize):

        tokenized_string = []

        toks = string_to_tokenize.split()

        for this_tok in toks:
            split_toks = self.tok_split(this_tok) # check if splits

            for tok in split_toks:
                r_stripped_toks = self.right_strip(tok)
                l_stripped_toks = self.left_strip(r_stripped_toks)
                tokenized_string.extend(l_stripped_toks)

        return " ".join(tokenized_string)

    @staticmethod
    def convert_to_training_format(conll_file_path):

        pass

if __name__ == "__main__":

    s = 'The "dog." is (here): to stay, I think. 47-52'

    tokenizer = NerTokenization()
    s2 = tokenizer.tokenize_string(s)
    print(s2)
