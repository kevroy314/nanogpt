"""Kevin Horecka's Implementation of https://www.youtube.com/watch?v=kCc8FmEb1nY"""
from abc import ABC
import torch # we use PyTorch: https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F

import plotext as plt
from tqdm.auto import tqdm

DEFAULT_FF_LAYER_SCALE_UP = 4
DEFAULT_LAYER_NORM_EPS = 1e-5

DEFAULT_H_DROPOUT_P = 0.2
DEFAULT_MH_DROPOUT_P = 0.2
DEFAULT_FF_DROPOUT_P = 0.2

DEFAULT_DATA_SPLIT = 0.1
DEFAULT_BLOCK_SIZE = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_ATTENTION_HEADS = 6
DEFAULT_NUM_ATTENTION_LAYERS = 6
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_LEARNING_RATE = 3e-4

DEFAULT_EVAL_INTERVAL = 500
DEFAULT_EVAL_ITERS = 200
DEFAULT_NUM_ITERS = 5000

class Tokenizer(ABC): # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    def __init__(self, training_text=None, name="default"): # pylint: disable=unused-argument
        self.name = name
        self.vocab = []
    # pylint: disable=missing-function-docstring
    def encode(self, text):
        raise NotImplementedError("encode() is not implemented")
    # pylint: disable=missing-function-docstring
    def decode(self, tokens):
        raise NotImplementedError("decode() is not implemented")

class CharacterTokenizer(Tokenizer): # pylint: disable=missing-class-docstring
    def __init__(self, training_text, name="character_tokenizer"):
        super().__init__(name)
        self.vocab = sorted(list(set(training_text)))
        self._stoi = { ch:i for i,ch in enumerate(self.vocab) }
        self._itos = { i:ch for i,ch in enumerate(self.vocab) }

    def encode(self, text):
        return [self._stoi[c] for c in text]
    def decode(self, tokens):
        return ''.join([self._itos[i] for i in tokens])

class DataSplitter(ABC): # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    def __init__(self, data, name="default_splitter"): # pylint: disable=unused-argument
        self.name = name
    # pylint: disable=missing-function-docstring
    def test_train_split(self, data, val_split=DEFAULT_DATA_SPLIT):
        raise NotImplementedError("test_train_split() is not implemented")

class SimpleTemporalSplitter(DataSplitter): # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    def __init__(self, name="simple_temporal_splitter"):
        self.name = name
        super().__init__(name)
    # pylint: disable=missing-function-docstring
    def test_train_split(self, data, val_split=DEFAULT_DATA_SPLIT):
        _n = int(val_split*len(data)) # first 90% will be train, rest val
        return data[:_n], data[_n:] # train, val

def load_data(
        input_data_filepath='input.txt',
        tokenizer_cls=CharacterTokenizer,
        splitter_cls=SimpleTemporalSplitter,
        default_block_size=DEFAULT_BLOCK_SIZE,
        default_batch_size=DEFAULT_BATCH_SIZE,
        default_device='cpu'
):
    """Load data for a transformer model.

    Args:
        input_data_filepath (str): Path to the input text file.
        tokenizer (Tokenizer): Tokenizer to use for encoding the text.
        splitter (DataSplitter): Data splitter to use for splitting the data.
        block_size (int): Size of blocks to be used for training.
        batch_size (int): Size of batches to be used for training.
        random_state (int): Seed for random number generator.

    Returns:
        tuple: A tuple containing the training and validation data batches.
            (trainX, trainY, valX, valY)
    """
    with open(input_data_filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    _tok = tokenizer_cls(text)

    data = torch.tensor(_tok.encode(text), dtype=torch.long)

    _splitter = splitter_cls(data)
    train_data, val_data = _splitter.test_train_split(data)

    def _batch_generator(
            train_or_val,
            _train_data,
            _val_data,
            _block_size=default_block_size,
            _batch_size=default_batch_size,
            _device=default_device
    ):
        data = _train_data if train_or_val == 'train' else _val_data
        ix = torch.randint(len(data) - _block_size, (_batch_size,))
        x = torch.stack([data[i:i+_block_size] for i in ix])
        y = torch.stack([data[i+1:i+_block_size+1] for i in ix])
        x, y = x.to(_device), y.to(_device)
        return x, y

    # pylint: disable=unnecessary-lambda-assignment
    _batch_generator_with_data_injection = \
        lambda train_or_val, block_size, batch_size: \
            _batch_generator(train_or_val, train_data, val_data, block_size, batch_size)

    return _batch_generator_with_data_injection, _tok, _splitter

class Head(nn.Module): # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    def __init__(self, head_size, embed_dim, block_dim, mode="encoder", dropout_p=DEFAULT_H_DROPOUT_P):
        super().__init__()
        self.head_size = head_size
        self.embed_dim = embed_dim
        self.mode = mode
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_dim, block_dim)))
        self.dropout = nn.Dropout(dropout_p)
    # pylint: disable=missing-function-docstring
    def forward(self, x):
        _, T, C = x.shape # pylint: disable=invalid-name

        k = self.key(x)   # (B, T, 16)
        q = self.query(x) # (B, T, 16)
        wei =  q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

        if self.mode == "encoder":
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module): # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    def __init__(self, num_heads, head_size, embed_dim, block_dim, mode="encoder", mh_dropout_p=DEFAULT_MH_DROPOUT_P, h_dropout_p=DEFAULT_H_DROPOUT_P):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [
                Head(
                    head_size,
                    embed_dim=embed_dim,
                    block_dim=block_dim,
                    mode=mode,
                    dropout_p=h_dropout_p
                ) for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=mh_dropout_p)

    # pylint: disable=missing-function-docstring
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForwardLayer(nn.Module): # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    def __init__(self, embed_dim, layer_scale_up=4, dropout_p=DEFAULT_FF_DROPOUT_P):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, layer_scale_up * embed_dim),
            nn.ReLU(),
            nn.Linear(layer_scale_up * embed_dim, embed_dim),
            nn.Dropout(p=dropout_p)
        )
    # pylint: disable=missing-function-docstring
    def forward(self, x):
        return self.net(x)


class LayerNormalization(nn.Module): # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    def __init__(self, dim, eps=DEFAULT_LAYER_NORM_EPS):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
    # pylint: disable=missing-function-docstring
    def forward(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(2, keepdim=True, unbiased=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        out = self.gamma * xhat + self.beta
        return out

class AttentionBlock(nn.Module): # pylint: disable=missing-class-docstring
    def __init__(self, num_heads, embed_dim, block_dim, mode="encoder", ff_layer_scale_up=DEFAULT_FF_LAYER_SCALE_UP, ff_dropout_p=DEFAULT_FF_DROPOUT_P, mh_dropout_p=DEFAULT_MH_DROPOUT_P, h_dropout_p=DEFAULT_H_DROPOUT_P):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            ("Embedding dimension must be divisible by number of ",
             f"heads {embed_dim} % {num_heads} = {embed_dim % num_heads}")

        self.ln1 = LayerNormalization(embed_dim)
        self.sa = MultiHeadAttention(
            num_heads,
            int(embed_dim/num_heads),
            embed_dim,
            block_dim,
            mode,
            mh_dropout_p=mh_dropout_p,
            h_dropout_p=h_dropout_p
        )
        self.ln2 = LayerNormalization(embed_dim)
        self.ff = FeedForwardLayer(embed_dim, layer_scale_up=ff_layer_scale_up, dropout_p=ff_dropout_p)
    # pylint: disable=missing-function-docstring
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module): # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    def __init__(
            self,
            vocab_size,
            block_dim,
            embed_dim=DEFAULT_EMBEDDING_DIM,
            num_attention_heads_per_layer=DEFAULT_NUM_ATTENTION_HEADS,
            num_attention_blocks=DEFAULT_BLOCK_SIZE,
            ff_layer_scale_up=DEFAULT_FF_LAYER_SCALE_UP,
            ff_dropout_p=DEFAULT_FF_DROPOUT_P,
            mh_dropout_p=DEFAULT_MH_DROPOUT_P,
            h_dropout_p=DEFAULT_H_DROPOUT_P
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_dim = block_dim
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_dim, embed_dim)
        self.attention_blocks = nn.Sequential(
            *[
                AttentionBlock(
                    num_attention_heads_per_layer,
                    embed_dim,
                    block_dim,
                    mode="encoder",
                    ff_layer_scale_up=ff_layer_scale_up,
                    ff_dropout_p=ff_dropout_p,
                    mh_dropout_p=mh_dropout_p,
                    h_dropout_p=h_dropout_p
                ) for _ in range(num_attention_blocks)
            ]
        )
        self.ln_final = LayerNormalization(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    # pylint: disable=missing-function-docstring
    def forward(self, idx, targets=None):
        B, T = idx.shape # pylint: disable=invalid-name
        # idx and targets are both (B,T) tensor of integers
        _tok_embeds = self.token_embeddings(idx) # (B,T,C)
        _pos_embeds = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = _tok_embeds + _pos_embeds # (B,T,C)
        x = self.attention_blocks(x)
        x = self.ln_final(x)
        _logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            _loss = None
        else:
            B, T, C = _logits.shape # pylint: disable=invalid-name
            _logits = _logits.view(B*T, C)
            targets = targets.view(B*T)
            _loss = F.cross_entropy(_logits, targets)

        return _logits, _loss
    # pylint: disable=missing-function-docstring
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            _logits, _loss = self.forward(idx[:, -self.block_dim:])
            # focus only on the last time step
            _logits = _logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(_logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


if __name__ == "__main__":
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    plt.theme('dark')

    torch.manual_seed(1337)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu' # pylint: disable=invalid-name
    device = 'cpu' # pylint: disable=invalid-name
    batch_size = DEFAULT_BATCH_SIZE # pylint: disable=invalid-name
    block_size = DEFAULT_BLOCK_SIZE # pylint: disable=invalid-name
    training_steps = DEFAULT_NUM_ITERS # pylint: disable=invalid-name
    eval_iters = DEFAULT_EVAL_ITERS # pylint: disable=invalid-name
    eval_interval = DEFAULT_EVAL_INTERVAL # pylint: disable=invalid-name
    learning_rate = DEFAULT_LEARNING_RATE # pylint: disable=invalid-name
    embedding_dimension_size = DEFAULT_EMBEDDING_DIM # pylint: disable=invalid-name
    num_attention_heads = DEFAULT_NUM_ATTENTION_HEADS
    num_attention_layers = DEFAULT_NUM_ATTENTION_LAYERS
    print("Loading data...")
    batch_generator, tok, splitter = load_data(default_device=device)

    print("Loading model with vocab size", len(tok.vocab), "...")
    model = BigramLanguageModel(
        len(tok.vocab),
        block_dim=block_size,
        embed_dim=embedding_dimension_size,
        num_attention_blocks=num_attention_layers,
        num_attention_heads_per_layer=num_attention_heads,
        ff_dropout_p=DEFAULT_FF_DROPOUT_P,
        mh_dropout_p=DEFAULT_MH_DROPOUT_P,
        h_dropout_p=DEFAULT_H_DROPOUT_P,
        ff_layer_scale_up=DEFAULT_FF_LAYER_SCALE_UP)
    m = model.to(device) # pylint: disable=invalid-name

    print("Performing test call...")
    logits, loss = m(*batch_generator('train', block_size, batch_size)) # pylint: disable=not-callable
    origina_loss = loss
    print(logits.shape)
    print(loss)

    @torch.no_grad()
    def estimate_loss(): # pylint: disable=missing-function-docstring
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = batch_generator(split, block_size, batch_size)
                _, _loss = model(x, y)
                losses[k] = _loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # pylint: disable=missing-function-docstring
    def sample_zeros(max_new_tokens=100, _device=device, _tok=tok, _m=m):
        return _tok.decode(
            _m.generate(
                idx=torch.zeros((1, 1),
                dtype=torch.long,
                device=_device
            ),
            max_new_tokens=max_new_tokens
        )[0].tolist())

    print("Sampling before training...")
    original_sample = sample_zeros()
    print(original_sample)

    print("Training...")
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    train_history = []
    val_history = []
    prog = tqdm(range(training_steps), total=training_steps)
    for steps in prog: # increase number of steps for good results...
        # sample a batch of data
        xb, yb = batch_generator('train', block_size, batch_size)
        # evaluate the loss
        logits, loss = m(xb, yb) # pylint: disable=not-callable
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if steps % eval_interval == 0:
            estimated_losses = estimate_loss()
            prog.set_description(
                f"step {steps}: train loss {estimated_losses['train']:.4f}, " + \
                f"val loss {estimated_losses['val']:.4f}"
            )
            train_history.append(estimated_losses['train'])
            val_history.append(estimated_losses['val'])
            plt.clt() # to clear the terminal
            plt.cld() # to clear the data only
            plt.plot(list(range(len(train_history))), train_history, label="train")
            plt.plot(list(range(len(val_history))), val_history, label="val")
            plt.show()
            print(f"Final losses: {estimated_losses}")

    print("Original loss: ", origina_loss.item())
    print("Original sample: ", original_sample)
    print("Sampling after training...")
    print(sample_zeros(500))
