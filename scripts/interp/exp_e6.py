"""exp_e6.py: the E6 plug-in (magnitude as a number vs magnitude as stakes inside Gemma) for launch_interp
--experiment e6. Laptop-only code lives in scripts/interp_e6_laptop.py, so it is not packed into the pod's create
request."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interp_e6_laptop import (add_args, after_step, configure, live, manifest, on_pod_info,  # noqa: E402,F401
                              report, setup, stub, summarize)
