#!/usr/bin/env python3
"""hooks.py (POD-SIDE): residual-stream capture, activation patching, steering.

Runs INSIDE the pinned vLLM image (torch 2.13, bf16). PyTorch forward hooks on
the Gemma 4 text decoder layers (model_io.decoder_layers). No extra frameworks.

A decoder block's forward returns the residual stream after the block, shape
(batch, seq, hidden). We hook each block's OUTPUT (post-block residual), which
is the standard "resid_post" readout used for the magnitude probe (E1), for
activation patching (E2), and to build/apply the steering vector (E3).

Three capabilities, all via register_forward_hook, all removable:
  - Capture:  record resid_post at chosen layers (optionally only chosen positions).
  - Patch:    overwrite resid_post at (layer, positions) with a stored tensor.
  - Steer:    add alpha * v to resid_post at (layer, positions).

Positions are given as absolute indices into the current sequence; the caller
is responsible for the end-aligned bookkeeping the minimal pairs need (the text
after the limit is identical, so positions are aligned from the end of the
prompt — plan E2(b)). Everything is exploratory.
"""
import torch                                   # pod-only


def _block_output_residual(output):
    """A decoder block returns either a Tensor or a tuple whose first item is it."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_block_output(output, new_resid):
    if isinstance(output, tuple):
        return (new_resid,) + tuple(output[1:])
    return new_resid


class HookManager:
    """Attach/detach forward hooks on a list of decoder layers.

    layers: the ModuleList from model_io.decoder_layers(model).
    All hooks are no-ops until configured, and every method returns self for
    chaining. Call .remove() (or use as a context manager) to detach.
    """

    def __init__(self, layers):
        self.layers = layers
        self._handles = []
        self._captures = {}          # layer_idx -> captured tensor (detached, cpu)
        self._capture_cfg = {}       # layer_idx -> positions or None (all)
        self._patch_cfg = {}         # layer_idx -> (positions tensor, value tensor)
        self._steer_cfg = {}         # layer_idx -> (positions or None, alpha, vector)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return self

    # --- configuration -----------------------------------------------------
    def capture(self, layer_indices, positions=None):
        """Record resid_post at these layers. positions: 1D LongTensor or None (all)."""
        for i in layer_indices:
            self._capture_cfg[int(i)] = positions
        return self

    def patch(self, layer_idx, positions, value):
        """Overwrite resid_post at layer_idx, rows `positions`, with `value`.

        positions: 1D LongTensor of sequence indices.
        value: tensor broadcastable to (len(positions), hidden) or (hidden,).
        """
        self._patch_cfg[int(layer_idx)] = (positions, value)
        return self

    def steer(self, layer_idx, vector, alpha=1.0, positions=None):
        """Add alpha*vector to resid_post at layer_idx (positions or all)."""
        self._steer_cfg[int(layer_idx)] = (positions, float(alpha), vector)
        return self

    # --- the captured data -------------------------------------------------
    def captured(self):
        """{layer_idx: tensor} recorded during the last forward (cpu, float32)."""
        return dict(self._captures)

    # --- wiring ------------------------------------------------------------
    def arm(self):
        """(Re)attach hooks for the current configuration. Clears old captures."""
        self.remove()
        self._captures = {}
        touched = set(self._capture_cfg) | set(self._patch_cfg) | set(self._steer_cfg)
        for idx in sorted(touched):
            self._handles.append(self.layers[idx].register_forward_hook(self._make_hook(idx)))
        return self

    def _make_hook(self, idx):
        def hook(_module, _inputs, output):
            resid = _block_output_residual(output)
            # Steer (add), then patch (overwrite), then capture — capture sees
            # the value the next layer actually receives.
            if idx in self._steer_cfg:
                positions, alpha, vec = self._steer_cfg[idx]
                v = vec.to(resid.dtype).to(resid.device)
                if positions is None:
                    resid = resid + alpha * v
                else:
                    pos = positions.to(resid.device)
                    resid = resid.clone()
                    resid[:, pos, :] = resid[:, pos, :] + alpha * v
            if idx in self._patch_cfg:
                positions, value = self._patch_cfg[idx]
                pos = positions.to(resid.device)
                val = value.to(resid.dtype).to(resid.device)
                resid = resid.clone()
                resid[:, pos, :] = val
            if idx in self._capture_cfg:
                positions = self._capture_cfg[idx]
                take = resid if positions is None else resid[:, positions.to(resid.device), :]
                self._captures[idx] = take.detach().to("cpu", torch.float32)
            return _replace_block_output(output, resid)
        return hook


class BatchPatcher:
    """Per-row patches in one batched forward: row r gets resid_post[layer][r, pos] = value.

    E2 runs many (layer, position-type) patches of the same target prompt at once, one per
    batch row. Positions are indices into the CURRENT forward's sequence (the suffix, when a
    prefix KV cache is reused). Active only between arm() and disarm(), so the decode steps
    that follow the patched prefill run unhooked.
    """

    def __init__(self, layers):
        self.layers = layers
        self._cfg = {}               # layer -> list of (row, positions LongTensor, value)
        self._handles = []

    def add(self, layer_idx, row, positions, value):
        self._cfg.setdefault(int(layer_idx), []).append((int(row), positions, value))
        return self

    def arm(self):
        self.disarm()
        for idx in sorted(self._cfg):
            self._handles.append(self.layers[idx].register_forward_hook(self._make_hook(idx)))
        return self

    def disarm(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return self

    def _make_hook(self, idx):
        def hook(_module, _inputs, output):
            resid = _block_output_residual(output).clone()
            for row, positions, value in self._cfg[idx]:
                pos = positions.to(resid.device)
                resid[row, pos, :] = value.to(resid.dtype).to(resid.device)
            return _replace_block_output(output, resid)
        return hook


class PrefillAdder:
    """E3 steering: add delta[r] to resid_post[layer][r, positions] in the prompt's forward only.

    Fires once, on the first forward whose sequence length is `seq_len` (the uncached part of
    the prompt); the decode steps (length 1) and anything after are untouched. delta: (B, P, H).
    """

    def __init__(self, layer_module, positions, delta, seq_len):
        self.positions, self.delta, self.seq_len = positions, delta, int(seq_len)
        self.fired = []
        self._h = layer_module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        resid = _block_output_residual(output)
        if self.fired or resid.shape[1] != self.seq_len:
            return output
        resid = resid.clone()
        pos = self.positions.to(resid.device)
        resid[:, pos, :] = resid[:, pos, :] + self.delta.to(resid.device, resid.dtype)
        self.fired.append(int(resid.shape[1]))
        return _replace_block_output(output, resid)

    def remove(self):
        self._h.remove()


class GenAdder:
    """E3b steering that follows the reply: prompt deltas on the prefill, then per generated token.

    prefill: the first forward whose length is `seq_len` gets prompt_delta (B, P, H) at `positions`.
    decode:  forward call j (length 1) carries generated token t = j - 1 as its input; it gets
             gen_delta[:, t] (B, T, H) while t < T, then tail (B, H) (zeros to stop steering).
    """

    def __init__(self, layer_module, positions, prompt_delta, seq_len, gen_delta, tail):
        self.positions, self.prompt_delta, self.seq_len = positions, prompt_delta, int(seq_len)
        self.gen_delta, self.tail = gen_delta, tail
        self.calls = []                                   # sequence lengths seen, for checks
        self._h = layer_module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        resid = _block_output_residual(output)
        n = int(resid.shape[1])
        if not self.calls:
            if n != self.seq_len:
                return output
            self.calls.append(n)
            resid = resid.clone()
            pos = self.positions.to(resid.device)
            resid[:, pos, :] = resid[:, pos, :] + self.prompt_delta.to(resid.device, resid.dtype)
            return _replace_block_output(output, resid)
        self.calls.append(n)
        t = len(self.calls) - 2                           # generated token index carried by this call
        add = self.gen_delta[:, t] if t < self.gen_delta.shape[1] else self.tail
        return _replace_block_output(output, resid + add.to(resid.device, resid.dtype)[:, None, :])

    def remove(self):
        self._h.remove()


def mean_resid(captures_a, captures_b):
    """v[layer] = mean(a) - mean(b) over the position axis, per layer.

    Each captures_* is {layer: (batch, pos, hidden)}; returns {layer: (hidden,)}.
    Used to build the (x1e6 - x1) steering vector for E3.
    """
    out = {}
    for layer in sorted(set(captures_a) & set(captures_b)):
        a = captures_a[layer].mean(dim=(0, 1))
        b = captures_b[layer].mean(dim=(0, 1))
        out[layer] = a - b
    return out
