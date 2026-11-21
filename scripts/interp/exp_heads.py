"""exp_heads.py: the heads plug-in (attention heads at layers 38-44, logit lens, attention to the digits, latch test)
for launch_interp --experiment heads. Laptop-only code lives in scripts/interp_heads_laptop.py, so it is not packed
into the pod's create request."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interp_heads_laptop import (add_args, after_step, configure, live, manifest, on_pod_info,  # noqa: E402,F401
                                 report, setup, stub, summarize)
