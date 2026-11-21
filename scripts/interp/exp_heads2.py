"""exp_heads2.py: the heads2 plug-in (where X57's key heads read the limit from; path patching; the buyers' heads)
for launch_interp --experiment heads2. Laptop-only code lives in scripts/interp_heads2_laptop.py, so it is not packed
into the pod's create request."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interp_heads2_laptop import (add_args, after_step, configure, live, manifest, on_pod_info,  # noqa: E402,F401
                                  report, setup, stub, summarize)
