from importlib.metadata import version

from bpl.dixon_coles import DixonColesMatchPredictor
from bpl.extended_dixon_coles import ExtendedDixonColesMatchPredictor
from bpl.neutral_dixon_coles import NeutralDixonColesMatchPredictor
from bpl.neutral_dixon_coles_WC import NeutralDixonColesMatchPredictorWC

try:
    __version__ = version("bpl-next")
except PackageNotFoundError:
    # package not installed, e.g. running from source
    __version__ = "unknown"
