
from pathlib import Path
import os

destination = Path(__file__).resolve().parents[1] / 'models/nltk_data'
destination.mkdir(parents=True, exist_ok=True)
os.environ['NLTK_DATA'] = str(destination)
import nltk
for resource in ['averaged_perceptron_tagger_eng', 'maxent_ne_chunker_tab', 'words', 'wordnet', 'vader_lexicon']:
    if not nltk.download(resource, download_dir=str(destination), raise_on_error=True):
        raise SystemExit(f'Failed to install {resource}')
