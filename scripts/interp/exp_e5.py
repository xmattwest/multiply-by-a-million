"""exp_e5.py: the E5 plug-in (the appraisal: where does Study 3's appraisal line take over inside Gemma?) for
launch_interp --experiment e5. Laptop-only code lives in scripts/interp_e5_laptop.py, so it is not packed into the
pod's create request."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interp_e5_laptop import (add_args, after_step, configure, live, manifest, on_pod_info,  # noqa: E402,F401
                              report, setup, stub, summarize)
