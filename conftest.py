"""
Pytest conftest — ensures ``flowsheet_graph`` imports when tests are
collected with bare ``pytest`` (which, unlike ``python -m pytest``,
does not add the current working directory to ``sys.path``).
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
