"""exp_e4.py: the E4 plug-in (the cure: does Q17's range sentence override the seller's x1e6 - x1 signal?) for
launch_interp --experiment e4. Laptop-only code lives in scripts/interp_e4_laptop.py, so it is not packed into the
pod's create request."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interp_e4_laptop import (add_args, after_step, configure, live, manifest, on_pod_info,  # noqa: E402,F401
                              report, setup, stub, summarize)
