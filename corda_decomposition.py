import torch
import torch.nn as nn
import torch.nn.functional as F

class CorDAAdapterParams(nn.Module):
    """
    Submodule to hold the low-rank adapter parameters for a single linear layer.
    These parameters are automatically registered as part of the parent module.
    """
    def __init__(self, down, diag, up):
        super().__init__()
        self.down = nn.Parameter(down)
        self.diag = nn.Parameter(diag)
        self.up   = nn.Parameter(up)

    def forward(self, x):
        # This submodule is only used to store parameters.
        return x

class CorDADecomposition:
    """
    Implements the CorDA decomposition:
      1. Registers forward hooks on each nn.Linear layer to collect their input activations.
      2. For each linear layer with collected activations, computes the covariance,
         performs SVD on (W @ Cov), and selects a low-rank subspace.
      3. Inserts an adapter submodule (holding the low-rank parameters) into each linear layer.
         The base weight is frozen and only the adapter submodules are trainable.
    """
    def __init__(self, model, adapter_rank=8, mode="knowledge_preserved",
                saliency_method="grad_norm", adapter_fraction=0.1):
        self.model = model
        self.adapter_rank = adapter_rank
        self.mode = mode
        self.saliency_method = saliency_method
        self.adapter_fraction = adapter_fraction
        self.activation_dict = {}
        
    def compute_saliency(self, samples_fn, device):
        saliencies = {}
        
        # Register gradient hooks
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                def hook(module, grad_input, grad_output, name=name):
                    if module.weight.grad is not None:
                        saliencies[name] = torch.norm(module.weight.grad).item()
                handles.append(module.register_full_backward_hook(hook))

        # Forward/backward pass
        self.model.train()
        xb, yb = samples_fn('train', 256, device, batch_size=32)
        _, loss = self.model(xb, yb)
        loss.backward()

        # Cleanup
        for h in handles:
            h.remove()
        self.model.zero_grad()
        
        return saliencies

    def hook_fn(self, module, inputs, outputs, name):
        # Get the first input tensor - matches pattern for nn.Linear which takes a single input tensor
        if inputs and isinstance(inputs[0], torch.Tensor):
            input_tensor = inputs[0]
            if name not in self.activation_dict:
                self.activation_dict[name] = []
            # Store a detached copy of the input tensor to avoid memory leaks
            self.activation_dict[name].append(input_tensor.view(-1, input_tensor.size(-1)).detach().cpu())
            print(f"[HOOK] Captured input for layer '{name}' with shape {input_tensor.shape}")

    def register_hooks(self):
        """Register forward hooks for each nn.Linear module in the model."""
        print("Registering forward hooks for each nn.Linear in the model:")
        hook_handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"  -> {name}")
                # Use a closure to capture the current name
                handle = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self.hook_fn(mod, inp, out, name) 
                )
                hook_handles.append(handle)
        return hook_handles

    def run_decomposition(self, samples_fn, device, context_window_size=256, batch_size=32, num_passes=50):
        """Run forward passes through the model to collect activations."""
        # Clear activation dictionary before starting
        self.activation_dict = {}
        
        # Register hooks and collect activations
        hook_handles = self.register_hooks()
        
        self.model.train()  # Ensure training mode so hooks fire
        print(f"Running {num_passes} forward passes to gather activations (batch_size={batch_size}).")
        
        for i in range(num_passes):
            xb, _ = samples_fn('train', context_window_size, device, batch_size=batch_size)
            _ = self.model(xb) # the hooks are not firing although the model is in training mode because 
            print(f"Forward pass {i+1}/{num_passes} complete.")
        
        # Remove hooks after collection
        for handle in hook_handles:
            handle.remove()
        
        # Check if any activations were collected
        if not self.activation_dict or all(len(v)==0 for v in self.activation_dict.values()):
            print("WARNING: No activations were collected for any layer. Check your hook registration!")
        else:
            collected = sum(1 for v in self.activation_dict.values() if len(v) > 0)
            print(f"Collected activations for {collected} linear layers.")
            
        # Compute saliency and select layers
        saliencies = self.compute_saliency(samples_fn, device)
        sorted_layers = sorted(saliencies.items(), key=lambda x: -x[1])
        num_selected = max(1, int(len(sorted_layers) * self.adapter_fraction))
        selected_layers = {k for k, v in sorted_layers[:num_selected]}
        
        self.compute_adapters(self.activation_dict, selected_layers)

    def compute_adapters(self, activation_dict, selected_layers):
        """
        For each linear layer with collected activations, perform SVD on W @ Cov and
        insert an adapter submodule with the low-rank update parameters.
        """
        with torch.no_grad():
            for n, module in self.model.named_modules():
                if n in selected_layers and isinstance(module, nn.Linear) and n in activation_dict and activation_dict[n]:
                    # Concatenate all collected activations for this layer
                    X = torch.cat(activation_dict[n], dim=0)  # (N, in_dim)
                    N = X.size(0)
                    if N > 2000:
                        idx = torch.randperm(N)[:2000]
                        X = X[idx]
                        N = 2000
                    
                    print(f"Computing adapter for layer '{n}' with {N} activation samples")
                    
                    # Compute covariance matrix
                    Cov = (X.t() @ X) / (N + 1e-6)  # (in_dim, in_dim)
                    W = module.weight.data  # (out_dim, in_dim)
                    W_prime = W @ Cov.to(W.device)  # (out_dim, in_dim)
                    
                    try:
                        U, S, Vt = torch.linalg.svd(W_prime, full_matrices=False)
                    except Exception as e:
                        print(f"[SVD error] Layer '{n}': {e}")
                        continue
                    
                    r = min(self.adapter_rank, S.size(0))
                    if self.mode == "knowledge_preserved":
                        start = S.size(0) - r
                        selected_indices = torch.arange(start, S.size(0), device=S.device)
                    else:
                        selected_indices = torch.arange(r, device=S.device)
                    
                    subU = U[:, selected_indices]    # (out_dim, r)
                    subS = S[selected_indices]       # (r,)
                    subVt = Vt[selected_indices, :]  # (r, in_dim)
                    
                    # Create adapter parameters
                    adapter_down = subVt.clone()
                    adapter_diag = subS.clone()
                    adapter_up = subU.t().clone()
                    
                    # Create adapter submodule
                    adapter_submodule = CorDAAdapterParams(adapter_down, adapter_diag, adapter_up)
                    
                    # Add the adapter to the module
                    module.weight.requires_grad = False
                    safe_name = n.replace('.', '_')
                    unique_submodule_name = f"corda_adapter_{safe_name}"
                    
                    if unique_submodule_name in module._modules:
                        del module._modules[unique_submodule_name]
                    
                    module.add_module(unique_submodule_name, adapter_submodule)
                    print(f"[CorDA] Inserted adapter submodule '{unique_submodule_name}' for layer '{n}'.")