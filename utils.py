import os
import requests
import torch

def download_shakespeare(file_path='input.txt'):
    if not os.path.exists(file_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(file_path, 'w') as f:
            f.write(requests.get(url).text)
    with open(file_path, 'r') as f:
        data = f.read()
    return data

def preprocess_data(data):
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = { ch: i for i,ch in enumerate(chars) }
    itos = { i: ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s]
    def decode(l):
        return ''.join([itos[i] for i in l])
    return vocab_size, encode, decode

def get_data_splits(encoded_data, split_ratio=0.9):
    n = len(encoded_data)
    train_data = encoded_data[:int(n*split_ratio)]
    val_data = encoded_data[int(n*split_ratio):]
    return train_data, val_data

def create_tensor_dataset(data_list):
    return torch.tensor(data_list, dtype=torch.long)

def build_get_batch_fn(train_tensor, val_tensor):
    def get_batch(split, context_window_size, device, batch_size=32):
        data = train_tensor if split == 'train' else val_tensor
        ix = torch.randint(len(data) - context_window_size, (batch_size,))
        x = torch.stack([data[i:i+context_window_size] for i in ix])
        y = torch.stack([data[i+1:i+context_window_size+1] for i in ix])
        return x.to(device), y.to(device)
    return get_batch
