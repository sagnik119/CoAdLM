import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_corda_adapter_if_any(linear_module, x):
    """
    Compute out = x @ W^T + b + adapter_update if adapter parameters exist.
    """
    # Compute base output using the linear module's forward method to trigger hooks
    out_base = linear_module(x)
    
    # Check for adapter submodules
    adapter_submodule = None
    for name, submodule in linear_module.named_modules():
        if "corda_adapter" in name and isinstance(submodule, nn.Module):
            adapter_submodule = submodule
            break
    
    if adapter_submodule is not None:
        # Apply adapter from the submodule
        out_adapter = x @ adapter_submodule.down.t()  # (B, r)
        out_adapter = out_adapter * adapter_submodule.diag  # element-wise multiply (B, r)
        out_adapter = out_adapter @ adapter_submodule.up  # (B, out_dim)
        out_base = out_base + out_adapter
    else:
        # Fall back to direct attributes if no submodule
        adapter_down = getattr(linear_module, "corda_adapter_down", None)
        adapter_diag = getattr(linear_module, "corda_adapter_diag", None)
        adapter_up = getattr(linear_module, "corda_adapter_up", None)
        
        if adapter_down is not None and adapter_diag is not None and adapter_up is not None:
            # Apply adapter
            out_adapter = x @ adapter_down.t()  # (B, r)
            out_adapter = out_adapter * adapter_diag  # element-wise multiply (B, r)
            out_adapter = out_adapter @ adapter_up  # (B, out_dim)
            out_base = out_base + out_adapter
    
    return out_base

class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.lin1 = nn.Linear(embed_size, 4 * embed_size)
        self.lin2 = nn.Linear(4 * embed_size, embed_size)
        self.act = nn.ReLU()

    def forward(self, x):
        out1 = apply_corda_adapter_if_any(self.lin1, x)
        out1 = self.act(out1)
        out2 = apply_corda_adapter_if_any(self.lin2, out1)
        return out2

class Head(nn.Module):
    def __init__(self, head_size, context_window_size, embed_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_window_size, context_window_size)))

    def forward(self, x):
        B, T, D = x.shape
        K = apply_corda_adapter_if_any(self.key, x)
        Q = apply_corda_adapter_if_any(self.query, x)
        V = apply_corda_adapter_if_any(self.value, x)
        scores = (Q @ K.transpose(-2, -1)) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, context_window_size, num_heads, head_size, embed_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, context_window_size, embed_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * embed_size, embed_size)

    def forward(self, x):
        out_cat = torch.cat([h(x) for h in self.heads], dim=-1)
        out = apply_corda_adapter_if_any(self.proj, out_cat)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, context_window_size, embed_size=384, num_heads=6):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        head_size = embed_size // num_heads
        self.mha = MultiHeadAttention(context_window_size, num_heads, head_size, embed_size)
        self.ffn = FeedForward(embed_size)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_window_size, embed_size=384, num_heads=6, n_layers=6):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(context_window_size, embed_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(context_window_size, embed_size, num_heads)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape
        if T > self.pos_emb.weight.size(0):
            token_ids = token_ids[:, -self.pos_emb.weight.size(0):]
            T = token_ids.size(1)
        tok = self.token_emb(token_ids)
        pos = self.pos_emb(torch.arange(T, device=token_ids.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = apply_corda_adapter_if_any(self.lm_head, x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=100):
        for _ in range(max_new_tokens):
            T = token_ids.size(1)
            if T > self.pos_emb.weight.size(0):
                token_ids_cond = token_ids[:, -self.pos_emb.weight.size(0):]
            else:
                token_ids_cond = token_ids
            logits, _ = self.forward(token_ids_cond)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids