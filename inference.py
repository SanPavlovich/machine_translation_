import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nltk.tokenize import word_tokenize
from prepare_data import clean_text

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NEC_TOKENS_LEN = 25
SOS_IND = 0
EOS_IND = 1
PAD_IND = 3

def greedy_translate(sentence, model, train_dataset):    
    for word in word_tokenize(sentence.lower()):
        if not word in train_dataset.x_tokenizer.tokens:
            print('unknown word:', word)
    
    sen2idx = [0] + train_dataset.x_tokenizer.convert_tokens_to_idx(word_tokenize(sentence.lower())) + [1]
    sen2idx = sen2idx + [3] * (NEC_TOKENS_LEN - len(sen2idx))

    model.eval()
    with torch.no_grad():
        outp = model(torch.Tensor(sen2idx).type(torch.long).unsqueeze(dim=0).to(device), None)

    return ' '.join(train_dataset.y_tokenizer.convert_idx_to_tokens(outp.squeeze().argmax(axis=1).cpu().numpy()))


def encode_part(model, x):
    batch_n = x.shape[0]
    length_tensor = (x != PAD_IND).sum(axis=1).cpu() # get seq lens to pack

    # encoder
    x = model.encoder_embed(x)
    x = pack_padded_sequence(x, length_tensor, batch_first=True, enforce_sorted=False)

    _, prev_state = model.encoder_lstm_layer(x)

    # reshape prev_state
    prev_state = (prev_state[0].reshape(model.lstm_layers, batch_n, -1),
                  prev_state[1].reshape(model.lstm_layers, batch_n, -1))

    return prev_state


class BeamSearchNode:
  def __init__(self, prev_state, prev_node, log_prob, token_idx, length):
      self.prev_state = (prev_state[0].squeeze(1), prev_state[1].squeeze(1))
      self.prev_node = prev_node
      self.token_idx = token_idx
      self.log_prob = log_prob
      self.length = length

  def eval(self):
      return self.log_prob
  
def get_next_nodes(model, prev_nodes, beam_width):
    # generate beam_width * len(prev_nodes) candidates for next iteration
    cur_nodes = []
    for prev_node in prev_nodes:
        prev_token_emb = model.decoder_embed(
            torch.tensor([prev_node.token_idx]).to(device)
            )

        out, prev_state = model.decoder_lstm_layer(prev_token_emb, prev_node.prev_state)
        decoder_output = torch.nn.functional.softmax(model.decoder_linear(out), dim=1)
        # choose top k tokens of the current decoder iteration for prev_node
        top_probs, indexes = torch.topk(decoder_output, beam_width)

        # create node for each combination prev_node and one of top k variant
        for top_val, token_idx in zip(top_probs.cpu().detach().numpy()[0], 
                                        indexes.cpu().detach().numpy()[0]):
            cur_node = BeamSearchNode(
                prev_state=prev_state,
                prev_node=prev_node,
                log_prob=prev_node.log_prob + np.log(top_val),
                token_idx=token_idx,
                length=prev_node.length + 1)
            cur_nodes.append(cur_node)

    return sorted(cur_nodes, key=lambda x: -x.log_prob)[:beam_width]


def beam_search_inference_for_texts_list(texts_list, model, x_tokenizer, y_tokenizer, beam_width):
    tokenised_text_list = []
    mx_seq_len = 0
    for text in texts_list:
        text = clean_text(text)
        x_text = x_tokenizer.convert_text_to_idx(text)
        tokenised_text_list.append(x_text)
        mx_seq_len = max(mx_seq_len, len(x_text))

    text_variants = []
    for tok_text in tokenised_text_list:
        x_pad_part = [x_tokenizer.pad_idx] * max(0, mx_seq_len - len(tok_text))

        x = torch.tensor([tok_text + x_pad_part]).to(device)
        prev_state = encode_part(model, x)

        # create <sos> tokens for each sentence
        prev_token_idx = torch.tensor([SOS_IND]).to(device)
        # beam_search start
        all_nodes = []
        prev_nodes = []
        end_nodes = []

        # create BeamSearchNode
        node = BeamSearchNode(
            prev_state=prev_state,
            prev_node=None,
            log_prob=0,
            token_idx=SOS_IND,
            length=1)
        all_nodes.append(node)
        prev_nodes.append(node)

        while True:
            cur_nodes = get_next_nodes(model, prev_nodes, beam_width)

            # collect end_nodes (if the node is <EOS> token and not first node)
            for cur_node in cur_nodes:
                if cur_node.token_idx == EOS_IND and cur_node.prev_node is not None:
                    end_nodes.append(cur_node)
            all_nodes += cur_nodes
            prev_nodes = cur_nodes
            # finish condition
            if len(end_nodes) >= beam_width or len(all_nodes) >= beam_width * NEC_TOKENS_LEN:
                break

        if len(end_nodes) < beam_width:
          end_nodes += cur_nodes[:beam_width - len(end_nodes)]
        most_likelihood_texts = []

        # collect the most probable texts
        for end_node in end_nodes:
            answer = []
            cur_node = end_node
            while cur_node is not None:
                answer.append(cur_node.token_idx)
                cur_node = cur_node.prev_node

            most_likelihood_texts.append(' '.join(y_tokenizer.convert_idx_to_tokens(answer[::-1])))
        text_variants.append(most_likelihood_texts)
    return text_variants

  

  