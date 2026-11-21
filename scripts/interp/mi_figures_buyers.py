"""mi_figures_buyers.py: the buyers session's figure plug-in for the research website (cards()).
Laptop-only code lives in scripts/interp_buyers_figures.py, so it is not packed into the pod's create request."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interp_buyers_figures import cards  # noqa: E402,F401
