"""scale-invariance interpretability track (pod-side + laptop driver).

Pod-side modules (model_io, hooks, tasks, run_interp) import torch/transformers
and run only on the pod inside the pinned vLLM image. The laptop driver
(scripts/launch_interp.py) is standard-library only and never imports them.
"""
