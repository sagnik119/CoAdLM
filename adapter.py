import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_corda_adapter_if_any(linear_module, x):
    """
    Compute out = x @ W^T + b + adapter_update if adapter parameters exist.
    """
    # Compute base output using the linear module's forward method to trigger hooks
    out_base = linear_module(x)
    
    # Check if adapter parameters exist
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