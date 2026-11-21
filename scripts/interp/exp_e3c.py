"""exp_e3c.py: the E3c layer-sweep plug-in (MI Session 1) for launch_interp --experiment e3c.
Laptop-only code lives in scripts/interp_e3c_laptop.py, so it is not packed into the pod's create request."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interp_e3c_laptop import (add_args, after_step, configure, live, manifest, on_pod_info,  # noqa: E402,F401
                               report, setup, stub, summarize)
