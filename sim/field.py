"""Generated competition fields. The original smooth course keeps its historical path."""
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / 'output'
DEFAULT_BEND = 'corner'
FIELD_DIRS = {'corner': OUTPUT / 'athletics_corner', 'smooth': OUTPUT / 'athletics'}
