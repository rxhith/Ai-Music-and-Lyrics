import torch
import torch.nn as nn
import torch.nn.functional as Functional
import numpy as np
import re
import string
import random

file = open("kanye_verses.txt", "r", encoding="utf8")
text = file.read()
text = text.replace("\n\n", "\n")


def clean_lyric(lyric):
    return re.sub("[^a-z' ]", "", lyric).replace("'", "")


lyrics = text.lower().split("\n")
lyrics = np.unique(lyrics)[1:].tolist()

cleaned_lyrics = [clean_lyric(lyric) for lyric in lyrics]
seq_size = 5


def create_sequences(lyric, seq_len):
    sequences = []

    if len(lyric.split()) <= seq_len:
        return [lyric]

    for itr in range(seq_len, len(lyric.split())):
        curr_seq = lyric.split()[itr - seq_len : itr + 1]
        sequences.append(" ".join(curr_seq))

    return sequences


raw_sequences = [create_sequences(lyric, seq_size) for lyric in cleaned_lyrics]

sequences = np.unique(np.array(sum(raw_sequences, []))).tolist()
uniq_words = np.unique(np.array(" ".join(sequences).split(" ")))
uniq_words_idx = np.arange(uniq_words.size)

word_to_idx = dict(zip(uniq_words.tolist(), uniq_words_idx.tolist()))
idx_to_word = dict(zip(uniq_words_idx.tolist(), uniq_words.tolist()))

vocab_size = len(word_to_idx)

x_word = []
y_word = []

for seq in sequences:
    if len(seq.split()) != seq_size + 1:
        continue
    x_word.append(" ".join(seq.split()[:-1]))
    y_word.append(" ".join(seq.split()[1:]))


def get_seq_idx(seq):
    return [word_to_idx[word] for word in seq.split()]


x_idx = np.array([get_seq_idx(word) for word in x_word])
y_idx = np.array([get_seq_idx(word) for word in y_word])
num_hidden = 256
num_layers = 4
embed_size = 200
drop_prob = 0.3
lr = 0.001
num_epochs = 20
batch_size = 32


class LyricLSTM(nn.Module):
    def __init__(self, num_hidden, num_layers, embed_size, drop_prob, lr):
        super().__init__()
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.lr = lr
        self.embedded = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, num_hidden, num_layers, dropout=drop_prob, batch_first=True
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(num_hidden, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedded(x)
        lstm_output, hidden = self.lstm(embedded, hidden)
        dropout_out = self.dropout(lstm_output).reshape(-1, self.num_hidden)
        out = self.fc(dropout_out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.num_layers, batch_size, self.num_hidden).zero_(),
            weight.new(self.num_layers, batch_size, self.num_hidden).zero_(),
        )
        return hidden


model = LyricLSTM(num_hidden, num_layers, embed_size, drop_prob, lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

model.train()


def get_next_batch(x, y, batch_size):
    for itr in range(batch_size, x.shape[0], batch_size):
        batch_x = x[itr - batch_size : itr, :]
        batch_y = y[itr - batch_size : itr, :]
        yield batch_x, batch_y


for epoch in range(num_epochs):
    hidden_layer = model.init_hidden(batch_size)

    for x, y in get_next_batch(x_idx, y_idx, batch_size):
        inputs = torch.from_numpy(x).type(torch.LongTensor)
        act = torch.from_numpy(y).type(torch.LongTensor)
        hidden_layer = tuple([layer.data for layer in hidden_layer])
        model.zero_grad()
        output, hidden = model(inputs, hidden_layer)
        loss = loss_func(output, act.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


def predict(model, tkn, hidden_layer):
    x = np.array([[word_to_idx[tkn]]])
    inputs = torch.from_numpy(x).type(torch.LongTensor)
    hidden = tuple([layer.data for layer in hidden_layer])
    out, hidden = model(inputs, hidden)
    prob = Functional.softmax(out, dim=1).data.numpy()
    prob = prob.reshape(prob.shape[1],)
    top_tokens = prob.argsort()[-3:][::-1]
    selected_index = top_tokens[random.sample([0, 1, 2], 1)[0]]
    return idx_to_word[selected_index], hidden


def generate(model, num_words, start_text):
    model.eval()
    hidden = model.init_hidden(1)
    tokens = start_text.split()
    for token in start_text.split():
        curr_token, hidden = predict(model, token, hidden)
    tokens.append(curr_token)
    for token_num in range(num_words - 1):
        token, hidden = predict(model, tokens[-1], hidden)
        tokens.append(token)
    return " ".join(tokens)


def filter_words(text):
    filtered_words = ["hate", "profanity_word2", "profanity_word3"]
    for word in filtered_words:
        text = text.replace(word, '*' * len(word))
    return text


def get_lyric(start_text, num_words):
    generated_text = generate(model, num_words, start_text.lower())
    filtered_text = filter_words(generated_text)
    return filtered_text


get_lyric("I hate you", 7)
