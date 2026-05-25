import yaml
from pathlib import Path

with open(Path(__file__).parent / "settings.yml") as f:
    _settings = yaml.safe_load(f)

TOPIC_SETTINGS = _settings["topic_modeling"]
SEMANTIC_SETTINGS = _settings["semantic_similarity"]
