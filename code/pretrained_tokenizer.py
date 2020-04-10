from transformers import AutoTokenizer
import re

# IMPORTANT: this tokenizer is compatible with train.lang1 and train.lang2 files only.
# The unaligned.en and unaligned.fr files must be pre-processed before they are used!
# The pre-processed unaligned.en and unaligned.fr files are available in either of these locations:
# /project/cq-training-1/project2/teams/team08/processed_unaligned
# \Google Drive\Machine_Translation_Project\processed data


class Tokenizer:
    ''' Encodes and decodes english and french tokens '''
    
    def __init__(self, language):
        super().__init__()
        
        # Maximum number of tokens per line
        self.MAX_LENGTH = 64
        
        if language == 'en':
            self.pretrained_tokenizer_path = "tokenizer_data_en/"
            self.cap_fn = lambda x: x
            self.uncap_fn = lambda x: x
        elif language == 'fr':
            self.pretrained_tokenizer_path = "tokenizer_data_fr/"
            self.cap_fn = self.__add_cap_tokens
            self.uncap_fn = self.__remove_cap_tokens
        else:
            raise Exception("Unsupported language")

    def encode(self, text):
        ''' Returns the encoded labels and the attention masks (to exclude PAD tokens) '''
        text = self.cap_fn(text)

        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_tokenizer_path, cache_dir=None)
        tokens = tokenizer.encode_plus(text, max_length=self.MAX_LENGTH, pad_to_max_length=True)

        return tokens

    def decode(self, tokens):
        ''' Returns text corresponding to the tokens passed. Compatible with .lang2 format. '''
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_tokenizer_path, cache_dir=None)
        text = tokenizer.decode(tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True)

        text = self.uncap_fn(text)

        return text

    def __add_cap_tokens(self, text):
        ''' Adds capitalization token '@' and lowercases the capital '''
        output = text.strip()
        positions = []
        for m in re.finditer(r'\b[A-ZÀ-ÖÙ-Ý]', output):
            positions += [m.start()]
        
        for idx in reversed(positions):
            output = output[:idx] + '@ ' + output[idx].lower() + output[(idx+1):]
            
        return output

    def __remove_cap_tokens(self, tokens):
        ''' Removes capitalization token '@' and uppercases the next character '''
        output = re.sub(' +', ' ', tokens).strip()
        positions = []
        for m in re.finditer(r'@', output):
            positions += [m.start()]
        
        for idx in reversed(positions):
            # Catch the case when sequence is terminated by @
            if idx + 2 >= len(output):
                output = output[:idx]
                continue
            output = output[:idx] + output[(idx+2)].upper() + output[(idx+3):]
            
        return output

# Sample usage
t_fr = Tokenizer(language='fr')
tokens = t_fr.encode("( DE ) Madame la Présidente , Monsieur le Commissaire , Mesdames et Messieurs , c' est un des signes de la pauvreté absolue de notre société que nous devions encore discuter de la question de l' égalité entre les hommes et les femmes .")
print(tokens)
text = t_fr.decode(tokens['input_ids'])
print(text)

t_en = Tokenizer(language='en')
tokens = t_en.encode("for the second phase of the trials we just had different sizes small medium large and extra - large it 's true")
print(tokens)
text = t_en.decode(tokens['input_ids'])
print(text)