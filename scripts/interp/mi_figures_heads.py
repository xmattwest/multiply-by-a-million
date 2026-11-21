"""mi_figures_heads.py: the heads session's figure plug-in for the research website (cards()).
Laptop-only code lives in scripts/interp_heads_figures.py, so it is not packed into the pod's create request."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interp_heads_figures import cards  # noqa: E402,F401
