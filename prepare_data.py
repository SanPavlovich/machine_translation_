from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import os

signs = '.,?!:; '
numbers = ''.join([str(i) for i in range(10)])
RUSSIAN_LETTERS_AND_SIGNS = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' + signs + numbers
ENGLISH_LETTERS_AND_SIGNS = ''.join([chr(i) for i in range(ord('a'), ord('z') + 1)]) + signs + numbers
SOS_TOKEN = "<SOS>" # start of sentence token
EOS_TOKEN = "<EOS>" # end of sentence token
UNK_TOKEN = "<UNK>" # unknown token
PAD_TOKEN = "<PAD>" # padding token

# read, shuffle, split and reorganise data
def split_data(dir, FULL_DATA_PATH, X_TRAIN_PATH, Y_TRAIN_PATH, X_VALID_PATH, Y_VALID_PATH, 
               MAX_TEXT_LEN, VALIDATION_PROP, RANDOM_STATE):
    
    with open(dir + FULL_DATA_PATH) as f:
        split_lines = [line.strip().split('\t') for line in f.readlines()]

    en_texts = [line[0] for line in split_lines]
    ru_texts = [line[1] for line in split_lines]

    en_train_texts, en_val_texts, ru_train_texts, ru_val_texts = train_test_split(
        en_texts,
        ru_texts,
        test_size=VALIDATION_PROP,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # filter data by russian text length
    with open(dir + X_TRAIN_PATH, 'w') as f:
        f.write('\n'.join([tt for i, tt in enumerate(en_train_texts) if len(word_tokenize(ru_train_texts[i])) < MAX_TEXT_LEN]))

    with open(dir + Y_TRAIN_PATH, 'w') as f:
        f.write('\n'.join([tt for i, tt in enumerate(ru_train_texts) if len(word_tokenize(tt)) < MAX_TEXT_LEN]))

    with open(dir + X_VALID_PATH, 'w') as f:
        f.write('\n'.join([tt for i, tt in enumerate(en_val_texts) if len(word_tokenize(ru_val_texts[i])) < MAX_TEXT_LEN]))

    with open(dir + Y_VALID_PATH, 'w') as f:
        f.write('\n'.join([tt for i, tt in enumerate(ru_val_texts) if len(word_tokenize(tt)) < MAX_TEXT_LEN]))


# clean text, only availaible symbols and lower text
def clean_text(text):
    text = text.lower().strip()
    res_text = ''.join([
        tt for tt in text if tt in RUSSIAN_LETTERS_AND_SIGNS or tt in ENGLISH_LETTERS_AND_SIGNS
      ])
    return res_text


# create our tokenizer is similar to different exist tokenizer
class Tokenizer:
  def __init__(self, texts, sos, eos, unk, pad, tokens=None):
    self.sos = sos
    self.eos = eos
    self.unk = unk
    self.pad = pad
    self.tokens = tokens

    self.create_tokenizer(texts)

  def create_tokenizer(self, texts):
    texts = [word_tokenize(clean_text(text)) for text in texts] # clean each text
    tokens = set([word for text in texts for word in text]) # get tokens (words)
    tokens = [t for t in tokens if t not in (self.sos, self.eos, self.unk) and len(t)]
    self.tokens = self.tokens or [self.sos, self.eos, self.unk, self.pad] + tokens

    self.token2idx = {token: i for i, token in enumerate(self.tokens)}
    self.idx2token = {i: token for i, token in enumerate(self.tokens)}
    self.unk_idx = self.token2idx[self.unk]
    self.sos_idx = self.token2idx[self.sos]
    self.eos_idx = self.token2idx[self.eos]
    self.pad_idx = self.token2idx[self.pad]

  def tokenize(self, text):
    #converts text to a list of tokens
    tokens = []
    for tok in word_tokenize(text):
      if tok in self.token2idx:
        tokens.append(tok)
      else:
        tokens.append(self.unk_idx)

    return [self.sos] + tokens + [self.eos]

  def convert_tokens_to_idx(self, tokens):
    # convert tokens list to idx list
    idx_list = [self.token2idx.get(tok, self.unk_idx) for tok in tokens]

    return idx_list

  def convert_text_to_idx(self, text, seq_len=None):
    # convert text to idx list
    tokens = self.tokenize(text)[:seq_len]

    return self.convert_tokens_to_idx(tokens)

  def convert_idx_to_tokens(self, idx_list):
    # convert idx list to tokens list
    ans = []
    for idx in idx_list:
      ans.append(self.idx2token[idx])

    return ans


# create tokenizer based on train_texts_path
def prepare_tokenizer(texts_path):
    with open(texts_path) as f:
        lines = [clean_text(line.strip()) for line in f.readlines()]

    return Tokenizer(lines, sos=SOS_TOKEN, eos=EOS_TOKEN, unk=UNK_TOKEN, pad=PAD_TOKEN)


class TranslationDataset(Dataset):
    def __init__(self, x_seq_path, 
                 x_tokenizer=None, x_seq_len=None, is_train=True, y_seq_path=None, y_tokenizer=None, y_seq_len=None):
        
        self.is_train = is_train
        with open(x_seq_path) as f:
          self.x_seq_list = [line.strip() for line in f.readlines()]

        self.x_tokenizer = x_tokenizer or prepare_tokenizer(x_seq_path)
        self.x_seq_len = x_seq_len

        if self.is_train:
          with open(y_seq_path) as f:
            self.y_seq_list = [line.strip() for line in f.readlines()]
          self.y_tokenizer = y_tokenizer or prepare_tokenizer(y_seq_path)
        else:
          self.y_seq_list = None
          self.y_tokenizer = None
        self.y_seq_len = y_seq_len

    def __getitem__(self, idx):
        # pad all sequences to fixed len (for x length equals x_seq_len, for y length equals y_seq_len)
        x = clean_text(self.x_seq_list[idx])
        x = self.x_tokenizer.convert_text_to_idx(x, seq_len=self.x_seq_len)
        x_pad_part = [self.x_tokenizer.pad_idx] * max(0, self.x_seq_len - len(x))
        x = x + x_pad_part

        x = torch.tensor(x)
        if self.is_train:
            y = clean_text(self.y_seq_list[idx])
            y = self.y_tokenizer.convert_text_to_idx(y, seq_len=self.y_seq_len)[1:]
            y_pad_part = [self.y_tokenizer.pad_idx] * max(0, self.y_seq_len - len(y))
            y = y + y_pad_part
            y = torch.tensor(y)

            return x, y
        return x

    def cnt_y_tokens(self):
      return len(self.y_tokenizer.tokens)

    def cnt_x_tokens(self):
      return len(self.x_tokenizer.tokens)

    def __len__(self):
        return len(self.x_seq_list)
