import torch
import torch.nn.functional as F
import math
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        num = p.numel()
        total_params += num
        if p.requires_grad:
            trainable_params += num
    return total_params, trainable_params, (total_params - trainable_params)

def approximate_transformer_flops(model, batch_size, seq_len, steps, fraction_trainable=1.0):
    """
    Approximates total FLOPs for training steps iterations on a transformer model.
    It uses the following approximations:
      - Self-Attention cost per layer: O(B * T * d_model^2 + B * T^2 * d_model)
      - Feed-Forward cost per layer: O(B * T * d_model * d_ff)
    with d_ff = 4 * d_model.
    The backward pass is approximated as 1.5x the forward pass plus an additional
    cost proportional to the fraction of trainable parameters.
    """
    n_layers = getattr(model, 'n_layers', 6)
    d_model = getattr(model, 'd_model', 384)
    d_ff = getattr(model, 'd_ff', d_model*4)
    n_heads = getattr(model, 'n_heads', 6)
    B = batch_size
    T = seq_len

    attn_flops_per_layer = B * T * (d_model ** 2) + B * (T ** 2) * d_model
    ff_flops_per_layer = B * T * d_model * d_ff
    forward_flops_per_layer = attn_flops_per_layer + ff_flops_per_layer

    forward_flops_per_iter = n_layers * forward_flops_per_layer

    activation_backprop_factor = 1.5
    param_grad_factor = 1.0 * fraction_trainable
    total_factor = 1.0 + activation_backprop_factor + param_grad_factor

    flops_per_iter = forward_flops_per_iter * total_factor
    total_flops = flops_per_iter * steps
    return total_flops

def train_baseline(model, optimizer, get_batch_fn, device,
                   num_iters=1000, context_window_size=256, eval_interval=200):
    start_time = time.time()
    model.train()
    for it in tqdm(range(num_iters), desc="Baseline Training"):
        if it % eval_interval == 0:
            print(f"[Baseline] Iteration {it}")
        xb, yb = get_batch_fn('train', context_window_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    total_time = time.time() - start_time
    return total_time

@torch.no_grad()
def evaluate_model(model, get_batch_fn, device, context_window_size=256, eval_iters=50):
    model.eval()
    total_loss = 0.0
    for _ in range(eval_iters):
        xb, yb = get_batch_fn('val', context_window_size, device)
        _, loss = model(xb, yb)
        total_loss += loss.item()
    avg_loss = total_loss / eval_iters
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def train_with_checkpoints(model, optimizer, get_batch_fn, device, num_iters, context_window_size, eval_interval, checkpoint_dir):
    loss_list = []
    start_time = time.time()
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()
    for it in tqdm(range(num_iters), desc="Training with Checkpoints"):
        xb, yb = get_batch_fn('train', context_window_size, device)
        _, loss = model(xb, yb)
        loss_list.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (it + 1) % eval_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{it+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint at iteration {it+1} to {ckpt_path}")
    total_time = time.time() - start_time
    return loss_list, total_time

def plot_loss(loss_list, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_list, marker='o')
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved loss plot to {filename}")

def load_latest_checkpoint(model, checkpoint_dir):
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not ckpt_files:
        print(f"No checkpoint found in {checkpoint_dir}.")
        return model
    ckpt_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    latest_ckpt = os.path.join(checkpoint_dir, ckpt_files[-1])
    model.load_state_dict(torch.load(latest_ckpt, map_location=model.token_emb.weight.device))
    print(f"Loaded checkpoint from {latest_ckpt}")
    return model

def finetune_corda_adapters(model, optimizer, get_batch_fn, device,
                            num_iters=500, context_window_size=256, eval_interval=100):
    start_time = time.time()
    model.train()
    for it in tqdm(range(num_iters), desc="CorDA Fine-Tuning"):
        if it % eval_interval == 0:
            print(f"[CorDA FT] Iteration {it}")
        xb, yb = get_batch_fn('train', context_window_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    total_time = time.time() - start_time
    return total_time

def finetune_full(model, optimizer, get_batch_fn, device,
                  num_iters=500, context_window_size=256, eval_interval=100):
    start_time = time.time()
    model.train()
    for it in tqdm(range(num_iters), desc="Full Fine-Tuning"):
        if it % eval_interval == 0:
            print(f"[Full FT] Iteration {it}")
        xb, yb = get_batch_fn('train', context_window_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    total_time = time.time() - start_time
    return total_time
