import random

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from attention import Attention

device = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN_SIZE = 256
LSTM_LAYERS = 3
INPUT_SIZE = 256
BATCH_SIZE = 32
IS_BIDIRECTIONAL_LAYERS = False
PAD_IND = 3
SOS_IND = 0
TEACHER_FORCING_RATIO = 0
NEC_TOKENS_LEN = 25


def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


class EncoderDecoderModel(nn.Module):
    # Simple Encoder-Decoder Model with 3 GRU Layers in encoder
    # And with 3 GRU in decoder, forward could be run with teacher_forcing
    def __init__(self, input_voc_size, cnt_classes, device=None,
                 input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, lstm_layers=LSTM_LAYERS, is_bidirectional=IS_BIDIRECTIONAL_LAYERS):
        
        super(EncoderDecoderModel, self).__init__()

        self.input_size = input_size
        self.lstm_layers = lstm_layers
        self.cnt_classes = cnt_classes
        self.hidden_size = hidden_size
        self.out_features = 1 if self.cnt_classes == 2 else self.cnt_classes
        self.is_bidirectional = is_bidirectional

        self.encoder_embed = nn.Embedding(input_voc_size, self.input_size)
        self.encoder_lstm_layer = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            bidirectional=self.is_bidirectional,
            batch_first=True
        )

        self.decoder_embed = nn.Embedding(self.cnt_classes, self.input_size)
        self.decoder_lstm_layer = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=(1 + self.is_bidirectional) * self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True
        )
        self.decoder_linear = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.out_features)

    def forward(self, x, target_val, teacher_forcing_ratio = TEACHER_FORCING_RATIO):
        batch_n = x.shape[0]
        length_tensor = (x != PAD_IND).sum(axis=1).cpu() # get seq lens to pack

        # encoder
        x = self.encoder_embed(x)
        x = pack_padded_sequence(x, length_tensor, batch_first=True, enforce_sorted=False)

        _, prev_state = self.encoder_lstm_layer(x)

        # reshape prev_state
        prev_state = (prev_state[0].reshape(self.lstm_layers, batch_n, -1),
                      prev_state[1].reshape(self.lstm_layers, batch_n, -1))

        # create <sos> tokens for each sentence
        prev_token_idx = torch.tensor([SOS_IND] * batch_n).reshape(-1, 1).to(device)

        # create probs for each token
        probs_tensor = None
        for i in range(NEC_TOKENS_LEN):
            if i != 0 and target_val is not None and random.random() < teacher_forcing_ratio:
                # teacher forcing
                prev_token_idx = target_val[:, i - 1].reshape(-1, 1)

            # decoder step
            prev_token_emb = self.decoder_embed(prev_token_idx)
            out, prev_state = self.decoder_lstm_layer(prev_token_emb, prev_state)

            out = self.decoder_linear(out)

            # token idx answer by greedy strategy
            prev_token_idx = out.argmax(axis=-1).reshape(batch_n, -1)

            # save probs for answers
            if probs_tensor is None:
                probs_tensor = out.unsqueeze(0)
            else:
                probs_tensor = torch.cat([probs_tensor, out.unsqueeze(0)])

            #if prev_token_idx.unique().shape[0] == 1 and prev_token_idx[0] == EOS_IND:
              #break

        return probs_tensor.transpose(1, 0)


class EncoderDecoderAttentionModel(nn.Module):
    # Simple Encoder-Decoder Model with 3 GRU Layers in encoder
    # And with 3 GRU in decoder, forward could be run with teacher_forcing
    def __init__(self, input_voc_size, cnt_classes, device=None, 
                 input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, lstm_layers=LSTM_LAYERS, is_bidirectional=IS_BIDIRECTIONAL_LAYERS):
        
        super(EncoderDecoderAttentionModel, self).__init__()

        self.input_size = input_size
        self.lstm_layers = lstm_layers
        self.cnt_classes = cnt_classes
        self.hidden_size = hidden_size
        self.out_features = 1 if self.cnt_classes == 2 else self.cnt_classes
        self.is_bidirectional = is_bidirectional

        self.encoder_embed = nn.Embedding(input_voc_size, self.input_size)
        self.encoder_lstm_layer = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            bidirectional=self.is_bidirectional,
            batch_first=True
        )
        self.attention = Attention(
            d_model=self.hidden_size,
            d_k=self.hidden_size,
            d_v=self.hidden_size,
            dropout=0
        )

        self.decoder_embed = nn.Embedding(self.cnt_classes, self.input_size)
        self.decoder_lstm_layer = nn.LSTM(
            input_size=2 * self.hidden_size, # concat attention and decoder hidden
            hidden_size=(1 + self.is_bidirectional) * self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True
        )
        self.decoder_linear = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.out_features)

    def forward(self, x, target_val, teacher_forcing_ratio = TEACHER_FORCING_RATIO):
        batch_n = x.shape[0]
        length_tensor = (x != PAD_IND).sum(axis=1).cpu() # get seq lens to pack

        ### encoder
        encod_emb = self.encoder_embed(x)
        # pack sequence
        packed_emb = pack_padded_sequence(encod_emb, length_tensor, batch_first=True, enforce_sorted=False)

        # encoder inference
        packed_out, enc_prev_state = self.encoder_lstm_layer(packed_emb)
        # enc_hiddens - hiddens for each iteration for last lstm layer
        enc_hiddens, _ = pad_packed_sequence(packed_out, batch_first=True)

        # reshape prev_state
        enc_prev_state = (enc_prev_state[0].reshape(self.lstm_layers, batch_n, -1),
                          enc_prev_state[1].reshape(self.lstm_layers, batch_n, -1))
        prev_state = enc_prev_state
        # create <sos> tokens for each sentence
        prev_token_idx = torch.tensor([SOS_IND] * batch_n).reshape(-1, 1).to(device)

        # create probs for each token
        probs_tensor = None
        for i in range(NEC_TOKENS_LEN):
            if i != 0 and target_val is not None and random.random() < teacher_forcing_ratio:
                # teacher forcing
                prev_token_idx = target_val[:, i - 1].reshape(-1, 1)

            # decoder step
            prev_token_emb = self.decoder_embed(prev_token_idx)
            # sum of decoder layers hiddens of iteration
            prev_dec_hidden = prev_state[0].sum(axis=0).unsqueeze(1)

            # attention between decoder and encoder
            out, _ = self.attention(prev_dec_hidden, enc_hiddens, enc_hiddens, return_attention=True)
            out = out.sum(axis=1) # sum for different attention heads

            # concat attention result and embeding of prev token
            decoder_input_embed = torch.cat([prev_token_emb, out], axis=-1)
            out, prev_state = self.decoder_lstm_layer(decoder_input_embed, prev_state)
            out = self.decoder_linear(out)

            # token idx answer by greedy strategy
            prev_token_idx = out.argmax(axis=-1).reshape(batch_n, -1)

            # save probs for answers
            if probs_tensor is None:
                probs_tensor = out.unsqueeze(0)
            else:
                probs_tensor = torch.cat([probs_tensor, out.unsqueeze(0)])

        return probs_tensor.transpose(1, 0)