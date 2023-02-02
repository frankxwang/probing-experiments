import math
from collections import OrderedDict

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Optional, Union, Iterator, cast, TypeVar, Generic, Callable

from torch.nn.modules.module import _addindent

import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from torch import nn

T = TypeVar("T")

class StaticModuleList(nn.ModuleList, Generic[T]):
    """ModuleList where the user vouches that it only contains objects of type T.
    This allows the static checker to work instead of only knowing that the contents are Modules.
    """

    # TBD lowpri: is it possible to do this just with signatures, without actually overriding the method bodies to add a cast?

    def __getitem__(self, index: int) -> T:
        return cast(T, super().__getitem__(index))

    def __iter__(self) -> Iterator[T]:
        return cast(Iterator[T], iter(self._modules.values()))

    def __repr__(self):
        # CM: modified from t.nn.Module.__repr__
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        modules = iter(self._modules.items())
        key, module = next(modules)
        n_rest = sum(1 for _ in modules)
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines + [f"+ {n_rest} more..."]

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


@dataclass(frozen=True)
class BertConfig:
    """Constants used throughout the Bert model. Most are self-explanatory.

    intermediate_size is the number of hidden neurons in the MLP (see schematic)
    type_vocab_size is only used for pretraining on "next sentence prediction", which we aren't doing.

    Note that the head size happens to be hidden_size // num_heads, but this isn't necessarily true and your code shouldn't assume it.
    """

    vocab_size: int = 28996
    intermediate_size: int = 3072
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_size: int = 64
    max_position_embeddings: int = 512
    dropout: float = 0.1
    type_vocab_size: int = 2
    layer_norm_epsilon: float = 1e-12


class BertSelfAttention(nn.Module):
    project_query: nn.Linear
    project_key: nn.Linear
    project_value: nn.Linear
    project_output: nn.Linear

    def __init__(self, config: BertConfig):
        "SOLUTION"
        super().__init__()
        self.num_heads = config.num_heads
        assert config.hidden_size % config.num_heads == 0
        # Note total head size can be smaller when we're doing tensor parallel and only some of the heads are in this module
        # But if it's larger then user probably forgot to specify head_size
        assert config.head_size * config.num_heads <= config.hidden_size, "Total head size larger than hidden_size"
        self.head_size = config.head_size
        self.project_query = nn.Linear(config.hidden_size, config.num_heads * self.head_size)
        self.project_key = nn.Linear(config.hidden_size, config.num_heads * self.head_size)
        self.project_value = nn.Linear(config.hidden_size, config.num_heads * self.head_size)
        self.project_output = nn.Linear(config.num_heads * self.head_size, config.hidden_size)

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)
        Return the attention pattern after scaling but before softmax.

        pattern[batch, head, q, k] should be the match between a query at sequence position q and a key at sequence position k.
        """
        "SOLUTION"
        b, s, h = x.shape
        q = self.project_query(x)
        q = rearrange(q, "b seq (head head_size) -> b head seq head_size", head=self.num_heads)
        # Does it feel more natural to split head and head_size along last two dims?
        k = self.project_key(x)
        k = rearrange(k, "b seq (head head_size) -> b head seq head_size", head=self.num_heads)
        out = einsum("b head seq_q head_size, b head seq_k head_size -> b head seq_q seq_k", q, k)
        # TBD lowpri: we can precompute 1/denominator and multiply it into Q before the einsum.
        # Could write exercise for this and see if we can detect a speed difference.
        out = out / (self.head_size**0.5)
        assert out.shape == (b, self.num_heads, s, s)
        return out

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        """
        additive_attention_mask: shape (batch, head=1, seq_q=1, seq_k) - used in training to prevent copying data from padding tokens. Contains 0 for a real input token and a large negative number for a padding token. If provided, add this to the attention pattern (pre softmax).

        Return: (batch, seq, hidden_size)
        """
        "SOLUTION"
        b, s, h = x.shape
        attention_pattern = self.attention_pattern_pre_softmax(x)
        if additive_attention_mask is not None:
            attention_pattern = attention_pattern + additive_attention_mask
        softmaxed_attention = attention_pattern.softmax(dim=-1)
        v = self.project_value(x)
        v = rearrange(v, "b seq (head head_size) -> b head seq head_size", head=self.num_heads)
        combined_values = einsum(
            "b head seq_k head_size, b head seq_q seq_k -> b head seq_q head_size",
            v,
            softmaxed_attention,
        )
        out = self.project_output(rearrange(combined_values, "b head seq head_size -> b seq (head head_size)"))
        assert out.shape == (b, s, h)
        return out

class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.batch_first = batch_first
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe[None, : x.size(1), :]
        else:
            x = x + self.pe[: x.size(0), None, :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_hid):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = BertSelfAttention(BertConfig(hidden_size=d_model, num_heads=nhead, head_size=d_model // nhead))

        self.linear1 = nn.Linear(d_model, d_hid)
        self.linear2 = nn.Linear(d_hid, d_model)
        self.activation = nn.ReLU()

    def forward(self, x, padding_mask):
        x = x + self.attn(x, padding_mask)
        x = x + self.mlp(x)
        return x

    def attn(self, x, padding_mask):
        x = self.norm1(x)
        additive_mask = torch.where(padding_mask, -10000, 0)[:, None, None, :]  # [batch, head=1, qpos=1, kpos]
        # print(additive_mask)
        x = self.self_attn(x, additive_attention_mask=additive_mask)
        return x

    def mlp(self, x):
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class ParenTransformer(nn.Module):
    def __init__(
        self, ntoken: int, nclasses: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.0
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers: StaticModuleList[TransformerBlock] = StaticModuleList(
            [TransformerBlock(d_model, nhead, d_hid) for _ in range(nlayers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.nhead = nhead

        self.decoder = nn.Linear(d_model, nclasses)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        padding_mask = x == SimpleTokenizer.PAD_TOKEN
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for l in self.layers:
            x = l(x, padding_mask)
        x = self.norm(x)
        x = self.decoder(x)
        return self.softmax(x[:, 0, :])

    def load_simple_transformer_state_dict(self, state_dict):
        new_dict = OrderedDict()
        for key, weight in state_dict.items():
            key = key.replace("transformer_encoder.", "").replace("out_proj", "project_output")
            if "in_proj_" in key:
                q, k, v = torch.tensor_split(weight, 3)
                # maps both in_proj_weight -> project_query.weight and in_proj_bias -> project_query.bias
                new_dict[key.replace("in_proj_", "project_query.")] = q
                new_dict[key.replace("in_proj_", "project_key.")] = k
                new_dict[key.replace("in_proj_", "project_value.")] = v
            else:
                if key == "pos_encoder.pe":
                    weight = weight[:, 0, :]  # remove extra dimension from posencoder due to earlier architechture
                new_dict[key] = weight
        self.load_state_dict(new_dict)


class SimpleTokenizer:
    START_TOKEN = 0
    PAD_TOKEN = 1
    END_TOKEN = 2
    base_d = {"[start]": START_TOKEN, "[pad]": PAD_TOKEN, "[end]": END_TOKEN}

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        # the 3 is because there are 3 special tokens (defined just above)
        self.t_to_i = {**{c: i + 3 for i, c in enumerate(alphabet)}, **self.base_d}
        self.i_to_t = {i: c for c, i in self.t_to_i.items()}

    def tokenize(self, strs: list[str], max_len: Optional[int] = None) -> torch.Tensor:
        def c_to_int(c: str) -> int:
            if c in self.t_to_i:
                return self.t_to_i[c]
            else:
                raise ValueError(c)

        if max_len is None:
            max_len = max((max(len(s) for s in strs), 1))

        ints = [
            [self.START_TOKEN] + [c_to_int(c) for c in s] + [self.END_TOKEN] + [self.PAD_TOKEN] * (max_len - len(s))
            for s in strs
        ]
        return torch.tensor(ints)

    def decode(self, tokens) -> list[str]:
        def int_to_c(c: int) -> str:
            if c < len(self.i_to_t):
                return self.i_to_t[c]
            else:
                raise ValueError(c)

        return [
            "".join(int_to_c(i.item()) for i in seq[1:] if i != self.PAD_TOKEN and i != self.END_TOKEN)
            for seq in tokens
        ]