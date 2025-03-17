import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (
    download_shakespeare, 
    preprocess_data, 
    get_data_splits, 
    create_tensor_dataset, 
    build_get_batch_fn
)
from model import TransformerLM
from train import (
    train_baseline, 
    evaluate_model, 
    finetune_corda_adapters, 
    finetune_full,
    count_parameters,
    approximate_transformer_flops,
    train_with_checkpoints,
    plot_loss,
    load_latest_checkpoint
)
from corda_decomposition import CorDADecomposition

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Create directories for checkpoints and figures.
    baseline_ckpt_dir = os.path.join("checkpoints", "baseline")
    corda_ckpt_dir = os.path.join("checkpoints", "corda")
    full_ckpt_dir = os.path.join("checkpoints", "full")
    fig_dir = "figures"
    os.makedirs(baseline_ckpt_dir, exist_ok=True)
    os.makedirs(corda_ckpt_dir, exist_ok=True)
    os.makedirs(full_ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Download & preprocess Shakespeare.
    raw_data = download_shakespeare()
    vocab_size, encode, decode = preprocess_data(raw_data)
    encoded_data = encode(raw_data)

    # 2. Create train/val splits and build get_batch.
    train_data, val_data = get_data_splits(encoded_data, split_ratio=0.9)
    train_tensor = create_tensor_dataset(train_data)
    val_tensor = create_tensor_dataset(val_data)
    get_batch = build_get_batch_fn(train_tensor, val_tensor)

    # 3. Create Baseline Model.
    model = TransformerLM(
        vocab_size=vocab_size,
        context_window_size=256,
        embed_size=384,
        num_heads=6,
        n_layers=6
    ).to(device)
    # Attach metadata.
    model.n_layers = 6
    model.d_model = 384
    model.d_ff = 384 * 4
    model.n_heads = 6

    BATCH_SIZE = 32
    SEQ_LEN = 256
    BASELINE_ITERS = 1000   # For testing, 10 iterations.
    FINE_TUNE_ITERS = 500
    ADAPTER_FRACTION = 0.01 # fraction of layers to adapt which rank highest in saliency

    #TODO: uncomment the below two paras for baseline training
    # total_params, trainable_params, _ = count_parameters(model)
    # baseline_flops = approximate_transformer_flops(model, BATCH_SIZE, SEQ_LEN, BASELINE_ITERS, fraction_trainable=1.0)
    # print(f"[Baseline Model] Total Params: {total_params:,}")
    # print(f"[Baseline Model] Trainable Params: {trainable_params:,}")
    # print(f"[Baseline Model] Estimated FLOPs: {baseline_flops/1e9:.2f} GFLOPs")

    # optimizer_baseline = optim.AdamW(model.parameters(), lr=1e-4)
    # print(f"Training Baseline Model for {BASELINE_ITERS} iterations...")
    model = load_latest_checkpoint(model, baseline_ckpt_dir)
    # baseline_loss_list, baseline_time = train_with_checkpoints(model, optimizer_baseline, get_batch, device, num_iters=BASELINE_ITERS, context_window_size=SEQ_LEN, eval_interval=100, checkpoint_dir=baseline_ckpt_dir)
    # print(f"Baseline training took {baseline_time:.2f} seconds")
    # plot_loss(baseline_loss_list, "Baseline Training Loss", os.path.join(fig_dir, "baseline_loss.png"))
    val_loss, val_ppl = evaluate_model(model, get_batch, device, SEQ_LEN, eval_iters=10)
    print(f"[Baseline] Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
    baseline_ckpt = model.state_dict()

    # ---------------------------
    # CorDA Decomposition.
    # ---------------------------
    def samples_fn(split, context_window_size, device, batch_size=32):
        return get_batch(split, context_window_size, device, batch_size)
    print("\nRunning CorDA decomposition (rank=8, knowledge_preserved)...")
    corda_decomp = CorDADecomposition(model, adapter_rank=8, mode="knowledge_preserved", saliency_method="grad_norm", adapter_fraction=ADAPTER_FRACTION) #TODO: make user defined choice arg for saliency method and fraction of layers to be adapted
    corda_decomp.run_decomposition(samples_fn, device, context_window_size=SEQ_LEN, batch_size=16)

    # ---------------------------
    # (A) Partial Fine-Tuning (CorDA).
    # ---------------------------
    print(f"\nCorDA Fine-Tuning Adapters for {FINE_TUNE_ITERS} iterations...")
    trainable_params_count = 0
    adapter_params = []

    # Find all adapter parameters
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Look for adapter submodules
            for subname, submodule in module.named_modules():
                if "corda_adapter" in subname:
                    for param_name, param in submodule.named_parameters():
                        param.requires_grad = True
                        adapter_params.append(param)
                        trainable_params_count += param.numel()

    print(f"Found {trainable_params_count} trainable adapter parameters across {len(adapter_params)} parameters")

    # Freeze all non-adapter parameters
    for name, param in model.named_parameters():
        if not any(param is adapter_param for adapter_param in adapter_params):
            param.requires_grad = False

    if trainable_params_count == 0:
        print("WARNING: No adapter parameters found. Adding dummy adapter to first linear layer.")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                in_features = module.weight.size(1)
                out_features = module.weight.size(0)
                rank = 8
                module.register_parameter("corda_adapter_down", nn.Parameter(torch.randn(rank, in_features)*0.01, requires_grad=True))
                module.register_parameter("corda_adapter_diag", nn.Parameter(torch.ones(rank)*0.01, requires_grad=True))
                module.register_parameter("corda_adapter_up", nn.Parameter(torch.randn(rank, out_features)*0.01, requires_grad=True))
                print(f"Added dummy adapter to {name}")
                break
            
    total_params_corda, trainable_params_corda, frozen_params_corda = count_parameters(model)
    fraction_trainable = trainable_params_corda / total_params_corda if total_params_corda > 0 else 0
    corda_flops = approximate_transformer_flops(model, BATCH_SIZE, SEQ_LEN, FINE_TUNE_ITERS, fraction_trainable=fraction_trainable, fraction_adapted=ADAPTER_FRACTION)
    print("[CorDA Partial FT] Param Stats:")
    print(f"  Total Params: {total_params_corda:,}")
    print(f"  Trainable:    {trainable_params_corda:,}")
    print(f"  Frozen:       {frozen_params_corda:,}")
    print(f"  Estimated FLOPs: {corda_flops/1e9:.2f} GFLOPs")
    adapter_optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model = load_latest_checkpoint(model, os.path.join("checkpoints", "corda"))
    corda_loss_list, corda_time = train_with_checkpoints(model, adapter_optimizer, get_batch, device, num_iters=FINE_TUNE_ITERS, context_window_size=SEQ_LEN, eval_interval=100, checkpoint_dir=os.path.join("checkpoints", "corda"))
    print(f"CorDA Fine-Tuning took {corda_time:.2f} seconds")
    plot_loss(corda_loss_list, "CorDA Fine-Tuning Loss", os.path.join(fig_dir, "corda_loss.png"))
    corda_val_loss, corda_val_ppl = evaluate_model(model, get_batch, device, SEQ_LEN, eval_iters=10)
    print(f"[CorDA-FineTuned] Val Loss: {corda_val_loss:.4f}, Val PPL: {corda_val_ppl:.2f}")
    print("Sample Generation after CorDA Fine-Tuning:")
    ctx = torch.zeros((1,1), dtype=torch.long, device=device)
    gen_tokens = model.generate(ctx, max_new_tokens=100)
    print(decode(gen_tokens[0].tolist()))

    # # ---------------------------
    # # (B) Full Fine-Tuning (All Params).
    # # ---------------------------
    # print(f"\nFull Fine-Tuning (All Params) for {FINE_TUNE_ITERS} iterations...")
    # full_model = TransformerLM(
    #     vocab_size=vocab_size,
    #     context_window_size=SEQ_LEN,
    #     embed_size=384,
    #     num_heads=6,
    #     n_layers=6
    # ).to(device)
    # full_model.load_state_dict(baseline_ckpt, strict=False) #TODO: make user defined choice arg for either this line or next
    # # full_model = load_latest_checkpoint(full_model, os.path.join("checkpoints", "full"))
    # full_model.n_layers = 6
    # full_model.d_model = 384
    # full_model.d_ff = 384*4
    # full_model.n_heads = 6
    # for p in full_model.parameters():
    #     p.requires_grad = True
    # total_params_full, trainable_params_full, frozen_params_full = count_parameters(full_model)
    # full_flops = approximate_transformer_flops(full_model, BATCH_SIZE, SEQ_LEN, FINE_TUNE_ITERS, fraction_trainable=1.0)
    # print("[Full Fine-Tuning] Param Stats:")
    # print(f"  Total Params: {total_params_full:,}")
    # print(f"  Trainable:    {trainable_params_full:,}")
    # print(f"  Frozen:       {frozen_params_full:,}")
    # print(f"  Estimated FLOPs: {full_flops/1e9:.2f} GFLOPs")
    # full_optimizer = optim.AdamW(full_model.parameters(), lr=1e-4)
    # full_loss_list, full_time = train_with_checkpoints(full_model, full_optimizer, get_batch, device, num_iters=FINE_TUNE_ITERS, context_window_size=SEQ_LEN, eval_interval=100, checkpoint_dir=os.path.join("checkpoints", "full"))
    # print(f"Full Fine-Tuning took {full_time:.2f} seconds")
    # plot_loss(full_loss_list, "Full Fine-Tuning Loss", os.path.join(fig_dir, "full_loss.png"))
    # full_val_loss, full_val_ppl = evaluate_model(full_model, get_batch, device, SEQ_LEN, eval_iters=10)
    # print(f"[Full Fine-Tuned] Val Loss: {full_val_loss:.4f}, Val PPL: {full_val_ppl:.2f}")

    print("\n---- Fine-Tuning Comparison ----")
    print(f"CorDA Partial Fine-Tuning => Val PPL: {corda_val_ppl:.2f}, FLOPs: {corda_flops/1e9:.2f} GFLOPs")
    # print(f"Full Fine-Tuning          => Val PPL: {full_val_ppl:.2f}, FLOPs: {full_flops/1e9:.2f} GFLOPs")

if __name__ == "__main__":
    main()
