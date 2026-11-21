"""exp_buyers.py: the buyers plug-in (MI Session 2) for launch_interp --experiment buyers.
Laptop-only code lives in scripts/interp_buyers_laptop.py, so it is not packed into the pod's create request."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interp_buyers_laptop import (add_args, after_step, configure, live, manifest, on_pod_info,  # noqa: E402,F401
                                  report, setup, stub, summarize)
