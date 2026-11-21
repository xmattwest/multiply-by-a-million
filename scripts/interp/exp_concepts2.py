"""exp_concepts2.py: the concepts follow-up (X63: is the six-concept part of the x1e6 - x1 direction what moves the price?
v split into that part and the rest, and the part ablated from the runs' own states) for launch_interp --experiment
concepts2. Laptop-only code lives in scripts/interp_concepts2_laptop.py; the pod code is tasks_concepts.py (POD_PLUGINS)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interp_concepts2_laptop import (POD_PLUGINS, add_args, after_step, configure, live, manifest,  # noqa: E402,F401
                                     on_pod_info, report, setup, stub, summarize)
