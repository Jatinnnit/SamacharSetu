"""Local, language-aware NLP. No submitted text is sent to external services."""
import json
import math
import os
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import regex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('NLTK_DATA', str(ROOT / 'models/nltk_data'))
LANGUAGES = {'en': 'English', 'hi': 'हिन्दी', 'mr': 'मराठी', 'bn': 'বাংলা', 'ta': 'தமிழ்', 'te': 'తెలుగు', 'gu': 'ગુજરાતી', 'kn': 'ಕನ್ನಡ', 'ml': 'മലയാളം', 'pa': 'ਪੰਜਾਬੀ', 'or': 'ଓଡ଼ିଆ', 'as': 'অসমীয়া'}
TOKEN_PATTERN = r"[\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)*|\p{N}+(?:[.,]\p{N}+)*|[^\s]"


def tokenize(text):
    return [{'text': m.group(), 'start': m.start(), 'end': m.end(),
             'is_word': bool(regex.search(r'[\p{L}\p{N}]', m.group()))}
            for m in regex.finditer(TOKEN_PATTERN, text)]


def sentences(text):
    return [x.strip() for x in regex.split(r'(?<=[.!?।॥])\s+|[\r\n]+', text) if x.strip()]


@lru_cache(maxsize=1)
def detector():
    from langid.langid import LanguageIdentifier, model
    return LanguageIdentifier.from_modelstring(model, norm_probs=True)


def detect(text):
    words = [t for t in tokenize(text) if t['is_word']]
    if not words or not regex.search(r'\p{L}', text):
        return {'language': 'und', 'confidence': None, 'uncertain': True, 'method': 'langid.py', 'notice': 'Add words to identify a language.'}
    lang, score = detector().classify(text)
    latin = len(regex.findall(r'\p{Script=Latin}', text))
    dev = len(regex.findall(r'\p{Script=Devanagari}', text))
    mixed = latin > 3 and dev > 3 and min(latin, dev) / max(latin + dev, 1) > .15
    uncertain = len(words) < 4 or float(score) < .75 or mixed
    return {'language': lang, 'confidence': round(float(score), 4), 'uncertain': uncertain,
            'mixed_script': mixed, 'method': 'Pretrained langid.py (97 languages)',
            'notice': 'Short or mixed-language input may be unreliable; choose the language explicitly.' if uncertain else 'Model confidence is not a calibrated guarantee; verify the language.'}


@lru_cache(maxsize=3)
def classifier(lang):
    file = ROOT / f'models/classifier-{lang}.joblib'
    if lang not in ('en', 'hi', 'mr') or not file.exists():
        return None
    # Only load bundled, trusted local model artifacts; never user-uploaded pickle files.
    return joblib.load(file)


def classify(text, lang):
    model = classifier(lang)
    if model is None:
        return {'status': 'unavailable', 'notice': 'No trained news classifier for this language.'}
    vec = model['features'].transform([text])
    if vec.nnz == 0:
        return {'status': 'uncertain', 'notice': 'No learned vocabulary matches this input.', 'labels': list(model['classifier'].classes_)}
    scores = model['classifier'].decision_function(vec)[0]
    classes = model['classifier'].classes_
    order = np.argsort(scores)[::-1]
    model_class = str(classes[order[0]])
    gap = float(scores[order[0]] - scores[order[1]])
    # A transparent heuristic, not a threshold chosen on test data or a probability.
    uncertain = gap < .35 or len(text.split()) < 3
    name_index = model['features'].get_feature_names_out()
    row = vec.tocsr()
    contribution = row.data * model['classifier'].coef_[order[0], row.indices]
    indices = np.argsort(contribution)[-8:][::-1]
    evidence = [{'feature': str(name_index[row.indices[i]]), 'contribution': round(float(contribution[i]), 4)} for i in indices if contribution[i] > 0]
    return {'status': 'uncertain' if uncertain else 'predicted', 'category': model_class,
            'score': round(float(scores[order[0]]), 4), 'margin_gap': round(gap, 4),
            'scores': [{'category': str(classes[i]), 'score': round(float(scores[i]), 4)} for i in order[:5]],
            'labels': list(classes), 'features': evidence,
            'notice': 'Uncalibrated SVM margin, not a probability. Categories are limited to this language’s training labels. Low-margin results need review.'}


def linguistics(text, lang):
    if lang != 'en':
        return {'status': 'unsupported', 'rows': [], 'entities': [], 'notice': 'POS, NER, stemming and lemmatization are available for English only in this build. Indic tokenization and trained classifiers remain available.'}
    import nltk
    from nltk.stem import SnowballStemmer, WordNetLemmatizer
    local_data = str(ROOT / 'models/nltk_data')
    if local_data not in nltk.data.path:
        nltk.data.path.insert(0, local_data)
    tokens = tokenize(text)
    try:
        tagged = nltk.pos_tag([t['text'] for t in tokens], lang='eng')
        tree = nltk.ne_chunk(tagged)
        stemmer, lemmatizer = SnowballStemmer('english'), WordNetLemmatizer()
        rows = []
        for token, (_, tag) in zip(tokens, tagged):
            group = tag[:1]
            wn_pos = {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}.get(group, 'n')
            description = {'N': 'Noun', 'V': 'Verb', 'J': 'Adjective', 'R': 'Adverb', 'D': 'Determiner', 'P': 'Pronoun', 'I': 'Preposition', 'C': 'Conjunction'}.get(group, 'Punctuation / other')
            rows.append({**token, 'tag': tag, 'description': description,
                         'stem': stemmer.stem(token['text']) if token['is_word'] else token['text'],
                         'lemma': lemmatizer.lemmatize(token['text'].lower(), wn_pos) if token['is_word'] else token['text']})
        entities, offset = [], 0
        for chunk in tree:
            if hasattr(chunk, 'label'):
                n = len(chunk.leaves())
                start, end = tokens[offset]['start'], tokens[offset + n - 1]['end']
                entities.append({'text': text[start:end], 'type': chunk.label(), 'start': start, 'end': end})
                offset += n
            else:
                offset += 1
        return {'status': 'available', 'rows': rows, 'entities': entities,
                'notice': 'English: NLTK perceptron POS, maximum-entropy NER, Snowball stemming and POS-aware WordNet lemmas. Entity predictions can be wrong; dates are not covered by this NER model.'}
    except LookupError:
        return {'status': 'unavailable', 'rows': [], 'entities': [], 'notice': 'English resources are missing. Run scripts/setup_nltk.py once, then restart the backend.'}


def summarize(text):
    parts = sentences(text)
    if len(parts) < 3 or len(text) < 180:
        return {'status': 'insufficient_text', 'sentences': [], 'notice': 'Too little source text for a useful summary. Read the available headline or excerpt below.'}
    try:
        x = TfidfVectorizer(analyzer='char', ngram_range=(3, 4)).fit_transform(parts)
        centroid = np.asarray(x.mean(axis=0))
        relevance = cosine_similarity(x, centroid).ravel()
        selected = sorted(np.argsort(relevance)[-min(3, max(1, len(parts) // 2)):].tolist())
        return {'status': 'available', 'sentences': [parts[i] for i in selected], 'notice': 'Extractive summary: sentences selected verbatim from the available text, with no added facts.'}
    except ValueError:
        return {'status': 'insufficient_text', 'sentences': [], 'notice': 'Not enough vocabulary for summarization.'}


STOPWORDS = {
    'en': {
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'is', 'are', 'was', 'were',
        'it', 'this', 'that', 'with', 'from', 'as', 'by', 'be', 'has', 'have', 'had', 'been', 'will',
        'would', 'can', 'could', 'about', 'after', 'all', 'also', 'their', 'they', 'them', 'who', 'which',
        'what', 'when', 'where', 'how', 'its', 'not', 'no', 'but', 'into', 'up', 'out', 'more', 'some',
        'said', 'says', 'say', 'one', 'two', 'new'
    },
    'hi': {
        'के', 'का', 'की', 'में', 'है', 'हैं', 'और', 'से', 'को', 'पर', 'ने', 'यह', 'वह', 'तो', 'ही', 'भी',
        'था', 'थे', 'थी', 'गया', 'गए', 'गई', 'किया', 'किए', 'करने', 'कर', 'रहा', 'रहे', 'रही', 'लिए',
        'दिया', 'इस', 'उस', 'एक', 'तक', 'हुआ', 'हुए', 'हुई', 'बाद', 'सकते', 'सकता', 'सकती', 'बहुत',
        'अपने', 'अपनी', 'अपने', 'कोई', 'कुछ', 'होने', 'होता', 'होते', 'होती', 'कहा', 'वाले', 'वाली'
    },
    'mr': {
        'आणि', 'व', 'या', 'आहे', 'आहेत', 'होता', 'होते', 'होती', 'नाही', 'ने', 'ना', 'चे', 'च्या', 'ची',
        'ला', 'मध्ये', 'वर', 'तून', 'कडून', 'पण', 'की', 'ते', 'ती', 'तो', 'हे', 'ह्या', 'त्या', 'त्यांच्या',
        'त्यांनी', 'एका', 'एक', 'झाला', 'झाली', 'झाले', 'केला', 'केली', 'केले', 'येथे', 'करणार', 'करून',
        'होणार', 'तसेच', 'मात्र', 'सर्व', 'काही', 'असे', 'अशी', 'असा', 'दिला', 'दिली', 'दिले'
    }
}


def extract_keywords(text, lang='auto', top_k=8):
    tokens = tokenize(text)
    words = [t['text'] for t in tokens if t['is_word']]
    if not words:
        return []
    lang_stops = STOPWORDS.get(lang, STOPWORDS['en'] | STOPWORDS['hi'] | STOPWORDS['mr'])
    filtered = []
    for w in words:
        cw = w.casefold().strip()
        if len(cw) >= 3 and cw not in lang_stops and not cw.isnumeric():
            filtered.append(w)
    if not filtered:
        return []
    counts = Counter(filtered)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    return [{'text': word, 'count': count} for word, count in ranked[:top_k]]


def analyze(text, language='auto'):
    detected = detect(text)
    lang = detected['language'] if language == 'auto' else language
    tokens = tokenize(text)
    prediction = classify(text, lang) if language != 'auto' or not detected['uncertain'] else {'status': 'uncertain', 'notice': 'Confirm the language to run classification on this short or mixed-language text.'}
    return {'language': lang, 'detection': detected, 'tokens': tokens,
            'sentences': sentences(text), 'counts': {'words': sum(t['is_word'] for t in tokens), 'tokens': len(tokens), 'sentences': len(sentences(text)), 'characters': len(text)},
            'linguistics': linguistics(text, lang), 'classification': prediction, 'summary': summarize(text),
            'sentiment': sentiment_emotion(text, lang), 'keywords': extract_keywords(text, lang)}


@lru_cache(maxsize=6)
def word_model(lang, mode):
    file = ROOT / f'models/word2vec-{lang}-{mode}.npz'
    if not file.exists():
        return None
    with np.load(file, allow_pickle=False) as data:
        words, vectors = data['words'].tolist(), data['vectors'].copy()
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return {'words': words, 'vectors': vectors, 'normalized': normalized, 'index': {w: i for i, w in enumerate(words)}}


def embeddings(lang, mode, word, other=''):
    model = word_model(lang, mode)
    if model is None:
        return {'status': 'unavailable', 'notice': 'No trained word embeddings for this language.'}
    words = model['words']
    word = unicodedata.normalize('NFC', word).casefold().strip()
    meta = json.loads((ROOT / f'models/word2vec-{lang}-{mode}.json').read_text())
    if word not in model['index']:
        return {'status': 'out_of_vocabulary', 'notice': 'This word is not in the learned vocabulary. Try one of the examples.', 'examples': words[:15], 'metadata': meta}
    other = unicodedata.normalize('NFC', other).casefold().strip()
    index = model['index'][word]
    scores = model['normalized'] @ model['normalized'][index]
    neighbours = [int(i) for i in np.argsort(scores)[::-1] if i != index][:8]
    return {'status': 'available', 'word': word, 'metadata': meta, 'vector': model['vectors'][index].tolist(),
            'nearest': [{'word': words[i], 'cosine': round(float(scores[i]), 4)} for i in neighbours],
            'pair_similarity': float(scores[model['index'][other]]) if other in model['index'] else None,
            'pair_notice': 'Second word not in vocabulary.' if other and other not in model['index'] else None}


def vector_analogy(lang, mode, pos_a, neg_b, pos_c):
    """Computes vector(pos_a) - vector(neg_b) + vector(pos_c) and returns the top nearest words."""
    model = word_model(lang, mode)
    if model is None:
        return {'status': 'unavailable', 'notice': 'No trained word embeddings for this language.'}
    words = model['words']
    a = unicodedata.normalize('NFC', pos_a).casefold().strip()
    b = unicodedata.normalize('NFC', neg_b).casefold().strip()
    c = unicodedata.normalize('NFC', pos_c).casefold().strip()

    missing = [orig for w, orig in [(a, pos_a), (b, neg_b), (c, pos_c)] if w not in model['index']]
    if missing:
        return {
            'status': 'out_of_vocabulary',
            'notice': f'Word(s) not in vocabulary: {", ".join(missing)}',
            'missing': missing,
            'examples': words[:15]
        }

    vec_a = model['normalized'][model['index'][a]]
    vec_b = model['normalized'][model['index'][b]]
    vec_c = model['normalized'][model['index'][c]]

    target = vec_a - vec_b + vec_c
    target_norm = np.linalg.norm(target)
    if target_norm > 1e-12:
        target = target / target_norm

    scores = model['normalized'] @ target
    exclude = {model['index'][a], model['index'][b], model['index'][c]}
    candidates = [int(i) for i in np.argsort(scores)[::-1] if i not in exclude][:6]

    return {
        'status': 'available',
        'query': {'pos_a': pos_a, 'neg_b': neg_b, 'pos_c': pos_c},
        'results': [{'word': words[i], 'cosine': round(float(scores[i]), 4)} for i in candidates],
        'method': f'Vector arithmetic: {pos_a} - {neg_b} + {pos_c} using {mode.upper()} 48-dim normalized embeddings'
    }



def coverage(articles):
    texts = [a['title'] + ' ' + (a.get('description') or '') for a in articles]
    matrix = [[None] * len(texts) for _ in texts]
    languages = {a['language'] for a in articles}
    for lang in languages:
        indices = [i for i, a in enumerate(articles) if a['language'] == lang]
        try:
            x = TfidfVectorizer(analyzer='char', ngram_range=(3, 5)).fit_transform([texts[i] for i in indices])
            sims = cosine_similarity(x)
            for i, a in enumerate(indices):
                for j, b in enumerate(indices):
                    matrix[a][b] = round(float(sims[i, j]), 4)
        except ValueError:
            pass
    word_sets = [set(t['text'].casefold() for t in tokenize(x) if t['is_word'] and len(t['text']) > 3) for x in texts]
    shared = sorted(set.intersection(*word_sets)) if word_sets else []
    entity_lists = [linguistics(t, a['language'])['entities'] for t, a in zip(texts, articles)]
    entity_sets = [{e['text'].casefold() for e in entities} for entities in entity_lists]
    return {'matrix': matrix, 'overlap': shared[:20], 'entities': entity_lists,
            'shared_entities': sorted(set.intersection(*entity_sets)) if entity_sets else [],
            'distinct_terms': [sorted(s - set(shared))[:12] for s in word_sets],
            'method': 'Character TF-IDF cosine similarity, fitted on the selected headlines and excerpts. Same-language matching only.',
            'notice': 'Textual overlap does not establish that stories describe the same event, nor does it measure truth or bias. Cross-language pairs are unavailable.'}


def ngram(training, evaluation, n=2, alpha=1.0):
    def words(text):
        return [t['text'].casefold() for t in tokenize(text) if t['is_word']]
    train_lines = [words(x) for x in sentences(training)]
    vocab = set(w for line in train_lines for w in line) | {'</s>', '<unk>'}
    counts, contexts = Counter(), Counter()
    for line in train_lines:
        seq = ['<s>'] * (n - 1) + line + ['</s>']
        for i in range(n - 1, len(seq)):
            ctx = tuple(seq[i - n + 1:i]) if n > 1 else ()
            counts[(ctx, seq[i])] += 1
            contexts[ctx] += 1
    logsum, events, zero = 0., [], False
    oov = 0
    for line in sentences(evaluation):
        raw = words(line)
        oov += sum(w not in vocab for w in raw)
        seq = ['<s>'] * (n - 1) + [w if w in vocab else '<unk>' for w in raw] + ['</s>']
        for i in range(n - 1, len(seq)):
            ctx = tuple(seq[i - n + 1:i]) if n > 1 else ()
            denom = contexts[ctx] + alpha * len(vocab)
            probability = (counts[(ctx, seq[i])] + alpha) / denom if denom else 0.
            if probability == 0:
                zero = True
            else:
                logsum += math.log(probability)
            events.append({'context': ' '.join(ctx), 'word': seq[i], 'count': counts[(ctx, seq[i])], 'context_count': contexts[ctx], 'probability': probability})
    perplexity = math.exp(-logsum / len(events)) if events and not zero else None
    sequence_prob = math.exp(logsum) if events and not zero else 0.0
    return {
        'n': n,
        'alpha': alpha,
        'vocabulary_size': len(vocab),
        'oov_count': oov,
        'events': events,
        'perplexity': perplexity,
        'sequence_probability': sequence_prob,
        'log_probability': round(logsum, 4) if events and not zero else None,
        'infinite': zero,
        'notice': 'Evaluation text is not fitted. Perplexity includes end-of-sentence events; unseen words map to <unk>. Compare only identical evaluation text and vocabulary.'
    }


def lsa(documents):
    try:
        # Use the mark-aware word tokenizer so Indic scripts are handled correctly.
        vectorizer = TfidfVectorizer(tokenizer=lambda s: [t['text'].casefold() for t in tokenize(s) if t['is_word']], token_pattern=None, max_features=2000)
        x = vectorizer.fit_transform(documents)
        k = min(3, x.shape[0] - 1, x.shape[1] - 1)
        if k < 1:
            return {'status': 'insufficient_data', 'notice': 'Add at least three documents with different words.'}
        svd = TruncatedSVD(n_components=k, random_state=42)
        coords = svd.fit_transform(x)
        terms = vectorizer.get_feature_names_out()
        topics = [{'component': i + 1, 'terms': [{'word': str(terms[j]), 'weight': round(float(component[j]), 4)} for j in np.argsort(abs(component))[-8:][::-1]]} for i, component in enumerate(svd.components_)]
        variance = np.nan_to_num(svd.explained_variance_ratio_, nan=0.).tolist()
        return {'status': 'available', 'topics': topics, 'coordinates': coords.tolist(), 'explained_variance': variance, 'notice': 'TF-IDF followed by truncated SVD (LSA). These are latent components, not labelled topics or LDA probabilities; signs are arbitrary.'}
    except ValueError:
        return {'status': 'insufficient_data', 'notice': 'The documents need more usable words.'}


EMOTION_LEXICON = {
    'joy': {'happy', 'joy', 'delight', 'glad', 'celebrat', 'win', 'victory', 'excel', 'cheer', 'success', 'love', 'pleas', 'great', 'proud', 'wonderful', 'inspiring', 'आनंद', 'आनंदी', 'उत्साह', 'हर्ष', 'जीत', 'विजय', 'सफलता', 'खुश'},
    'sadness': {'sad', 'sorrow', 'grief', 'tragic', 'depress', 'mourn', 'loss', 'fail', 'pain', 'tear', 'disappoint', 'heartbreak', 'दुःख', 'दुख', 'शोक', 'निराशा', 'हताश'},
    'anger': {'anger', 'angry', 'rage', 'furi', 'protest', 'outrage', 'hostil', 'hate', 'clash', 'scandal', 'corrupt', 'awful', 'terrible', 'frustrat', 'राग', 'क्रोध', 'गुस्सा', 'संताप', 'आक्रोश'},
    'fear': {'fear', 'scare', 'threat', 'danger', 'alarm', 'warn', 'panic', 'terror', 'crisis', 'risk', 'harm', 'devastat', 'भीती', 'भय', 'धोका', 'संकट', 'चिंता'},
    'surprise': {'surprise', 'shock', 'unexpect', 'astonish', 'amaze', 'stun', 'revelat', 'miracle', 'आश्चर्य', 'धक्का', 'चमत्कार'},
    'trust': {'trust', 'hope', 'confiden', 'faith', 'assur', 'promis', 'cooperat', 'peace', 'accord', 'helpful', 'reliable', 'विश्वास', 'आशा', 'शांती', 'मदत', 'सहकार्य'}
}


def sentiment_emotion(text, lang='en'):
    import nltk
    local_data = str(ROOT / 'models/nltk_data')
    if local_data not in nltk.data.path:
        nltk.data.path.insert(0, local_data)
    words = [t['text'].casefold() for t in tokenize(text) if t['is_word']]
    if not words:
        return {'status': 'insufficient_text', 'label': 'Neutral', 'compound': 0.0,
                'polarity': {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0},
                'emotions': {emo: 0.0 for emo in EMOTION_LEXICON}, 'dominant_emotion': 'Neutral',
                'evidence': {}, 'notice': 'Enter text with words to analyze sentiment.'}
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
    except Exception:
        scores = {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}

    compound = round(float(scores['compound']), 4)
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'

    emotion_counts = {emo: 0 for emo in EMOTION_LEXICON}
    evidence_words = {emo: [] for emo in EMOTION_LEXICON}
    for w in words:
        for emo, keywords in EMOTION_LEXICON.items():
            if any(w.startswith(k) or k in w for k in keywords):
                emotion_counts[emo] += 1
                if w not in evidence_words[emo]:
                    evidence_words[emo].append(w)

    total_emo = sum(emotion_counts.values())
    if total_emo > 0:
        emotion_dist = {emo: round(count / total_emo, 4) for emo, count in emotion_counts.items()}
        dominant_emo = max(emotion_dist.items(), key=lambda x: x[1])[0].capitalize()
    else:
        emotion_dist = {emo: 0.0 for emo in EMOTION_LEXICON}
        dominant_emo = 'Neutral'

    return {
        'status': 'available',
        'label': label,
        'compound': compound,
        'polarity': {k: round(float(v), 4) for k, v in scores.items()},
        'emotions': emotion_dist,
        'dominant_emotion': dominant_emo,
        'evidence': {k: v for k, v in evidence_words.items() if v},
        'notice': 'Sentiment via VADER lexicon; emotion intensities mapped across 6 emotional axes.'
    }


def opinion_mining(documents):
    cleaned = [d.strip() for d in documents if d.strip()]
    if len(cleaned) < 3:
        return {'status': 'insufficient_data', 'notice': 'Provide at least 3 opinion texts to perform topic-based opinion mining.'}

    analyzed_opinions = []
    for doc in cleaned:
        sent = sentiment_emotion(doc)
        analyzed_opinions.append({
            'text': doc,
            'label': sent['label'],
            'compound': sent['compound'],
            'dominant_emotion': sent['dominant_emotion'],
            'emotions': sent['emotions']
        })

    try:
        vectorizer = TfidfVectorizer(tokenizer=lambda s: [t['text'].casefold() for t in tokenize(s) if t['is_word']], token_pattern=None, max_features=1500)
        x = vectorizer.fit_transform(cleaned)
        k = min(3, len(cleaned) - 1, x.shape[1] - 1)
        if k < 1:
            k = 1
        svd = TruncatedSVD(n_components=k, random_state=42)
        coords = svd.fit_transform(x)
        terms = vectorizer.get_feature_names_out()
    except Exception:
        coords = np.zeros((len(cleaned), 1))
        terms = np.array([])
        k = 1

    topic_groups = defaultdict(list)
    for idx, (doc_info, coord) in enumerate(zip(analyzed_opinions, coords)):
        dominant_topic = int(np.argmax(np.abs(coord))) if len(coord) > 0 else 0
        topic_groups[dominant_topic].append(doc_info)

    topic_summaries = []
    for t_idx in range(k):
        docs_in_topic = topic_groups.get(t_idx, [])
        if not docs_in_topic:
            continue
        if len(terms) > 0 and t_idx < len(svd.components_):
            comp = svd.components_[t_idx]
            top_kw_idx = np.argsort(np.abs(comp))[-6:][::-1]
            keywords = [str(terms[j]) for j in top_kw_idx]
        else:
            keywords = ['general']

        pos_count = sum(1 for d in docs_in_topic if d['label'] == 'Positive')
        neg_count = sum(1 for d in docs_in_topic if d['label'] == 'Negative')
        neu_count = sum(1 for d in docs_in_topic if d['label'] == 'Neutral')
        total = len(docs_in_topic)
        avg_compound = round(float(np.mean([d['compound'] for d in docs_in_topic])), 4)

        emos = Counter(d['dominant_emotion'] for d in docs_in_topic if d['dominant_emotion'] != 'Neutral')
        topic_emotion = emos.most_common(1)[0][0] if emos else 'Neutral'

        topic_summaries.append({
            'topic_id': t_idx + 1,
            'keywords': keywords,
            'count': total,
            'positive_pct': round((pos_count / total) * 100, 1),
            'negative_pct': round((neg_count / total) * 100, 1),
            'neutral_pct': round((neu_count / total) * 100, 1),
            'avg_polarity': avg_compound,
            'sentiment_stance': 'Mostly Positive' if avg_compound >= 0.1 else ('Mostly Negative' if avg_compound <= -0.1 else 'Mixed / Neutral'),
            'dominant_emotion': topic_emotion,
            'opinions': docs_in_topic
        })

    overall_pos = sum(1 for d in analyzed_opinions if d['label'] == 'Positive')
    overall_neg = sum(1 for d in analyzed_opinions if d['label'] == 'Negative')
    overall_neu = sum(1 for d in analyzed_opinions if d['label'] == 'Neutral')
    total_all = len(analyzed_opinions)

    return {
        'status': 'available',
        'summary': {
            'total_opinions': total_all,
            'positive_count': overall_pos,
            'negative_count': overall_neg,
            'neutral_count': overall_neu,
            'overall_positive_pct': round((overall_pos / total_all) * 100, 1),
            'overall_negative_pct': round((overall_neg / total_all) * 100, 1),
            'topics_identified': len(topic_summaries)
        },
        'topics': topic_summaries,
        'notice': 'Topic-based opinion mining: LSA extracts latent discussion themes; VADER & emotion lexicons score polarity per topic.'
    }


CLASS_PROTOTYPES = {
    'en': {
        'auto': ['car', 'ev', 'suv', 'cars', 'hyundai', 'maruti', 'vehicle', 'drive', 'engine'],
        'business': ['sebi', 'rbi', 'adani', 'crypto', 'ipo', 'gst', 'market', 'bank', 'stocks', 'economy'],
        'education': ['iit', 'exam', 'education', 'students', 'schools', 'neet', 'college', 'university', 'study'],
        'elections': ['polls', 'election', 'bjp', 'telangana', 'opposition', 'ec', 'congress', 'vote', 'candidate'],
        'entertainment': ['actor', 'film', 'bts', 'singer', 'films', 'kapoor', 'cinema', 'movie', 'star'],
        'health': ['covid', 'omicron', 'covaxin', 'coronavirus', 'health', 'cases', 'hospital', 'doctor', 'vaccine'],
        'india': ['sc', 'case', 'nia', 'ed', 'cji', 'court', 'delhi', 'mumbai', 'police', 'state'],
        'lifestyle': ['fashion', 'style', 'travel', 'food', 'diet', 'beauty', 'wellness', 'trend'],
        'science': ['study', 'nasa', 'isro', 'space', 'earth', 'heat', 'moon', 'planet', 'research'],
        'sports': ['ipl', 'cup', 'vs', 'odi', 'dhoni', 'kohli', 'cricket', 'match', 'win', 'goal', 'team'],
        'technology': ['ai', 'apple', 'pro', 'oneplus', 'google', 'meta', 'phone', 'software', 'tech', 'device'],
        'world': ['us', 'pm', 'biden', 'breaking', 'china', 'un', 'russia', 'ukraine', 'war', 'global']
    },
    'hi': {
        'business': ['बाजार', 'शेयर', 'बैंक', 'कंपनी', 'रुपये', 'कारोबार'],
        'entertainment': ['फिल्म', 'अभिनेता', 'सिनेमा', 'गाना', 'स्टार', 'मूवी'],
        'sports': ['क्रिकेट', 'मैच', 'टीम', 'खिलाड़ी', 'कप', 'जीत'],
        'technology': ['तकनीक', 'मोबाइल', 'इंटरनेट', 'फोन', 'एप'],
        'state': ['सरकार', 'पुलिस', 'मामला', 'प्रशासन', 'राज्य']
    },
    'mr': {
        'business': ['बाजार', 'उद्योग', 'बँक', 'कंपनी', 'पैसा', 'अर्थव्यवस्था'],
        'entertainment': ['चित्रपट', 'अभिनेता', 'कलाकार', 'नाटक', 'गाणे'],
        'sports': ['क्रिकेट', 'सामना', 'खेळाडू', 'विजय', 'स्पर्धा'],
        'technology': ['तंत्रज्ञान', 'मोबाइल', 'यंत्र', 'फोन'],
        'state': ['सरकार', 'पोलीस', 'प्रशासन', 'राज्य', 'मंत्री']
    }
}


def classify_with_embeddings(text, lang='en', mode='skipgram'):
    model = word_model(lang, mode)
    if model is None:
        return {'status': 'unavailable', 'notice': 'No word vector model for this language.'}
    words_info = tokenize(text)
    clean_words = []
    for t in words_info:
        if t['is_word']:
            clean_words.append(unicodedata.normalize('NFC', t['text']).casefold().strip())

    pos_data = []
    if lang == 'en':
        ling = linguistics(text, 'en')
        if ling['status'] == 'available':
            for r in ling['rows']:
                pos_data.append({
                    'text': r['text'],
                    'tag': r['tag'],
                    'description': r['description'],
                    'in_vocab': unicodedata.normalize('NFC', r['text']).casefold().strip() in model['index']
                })

    known_vectors = []
    used_words = []
    for w in clean_words:
        if w in model['index']:
            known_vectors.append(model['vectors'][model['index'][w]])
            used_words.append(w)

    if not known_vectors:
        return {
            'status': 'out_of_vocabulary',
            'predicted_category': None,
            'similarity_score': 0.0,
            'scores': [],
            'words_used': [],
            'pos_tokens': pos_data,
            'notice': 'None of the words in this text are in the learned Word2Vec vocabulary.'
        }

    doc_vec = np.mean(known_vectors, axis=0)
    norm = np.linalg.norm(doc_vec)
    if norm > 1e-12:
        doc_vec = doc_vec / norm

    prototypes = CLASS_PROTOTYPES.get(lang, CLASS_PROTOTYPES['en'])
    scores = []
    for cat, proto_words in prototypes.items():
        proto_vecs = [model['vectors'][model['index'][pw]] for pw in proto_words if pw in model['index']]
        if proto_vecs:
            p_vec = np.mean(proto_vecs, axis=0)
            p_norm = np.linalg.norm(p_vec)
            if p_norm > 1e-12:
                p_vec = p_vec / p_norm
            sim = float(doc_vec @ p_vec)
            scores.append({'category': cat, 'similarity': round(sim, 4)})

    scores.sort(key=lambda s: s['similarity'], reverse=True)
    predicted = scores[0]['category'] if scores else 'unknown'

    return {
        'status': 'available',
        'predicted_category': predicted,
        'similarity_score': scores[0]['similarity'] if scores else 0.0,
        'scores': scores[:6],
        'words_used': used_words,
        'pos_tokens': pos_data[:20],
        'vector_sample': [round(float(v), 4) for v in doc_vec[:8]],
        'dimensions': len(doc_vec),
        'method': f'Averaged {len(used_words)} word embeddings ({mode.upper()}) + cosine similarity to category centroids'
    }


def compute_tfidf(documents, max_features=25):
    """
    Computes Term Frequency (TF), Inverse Document Frequency (IDF),
    and TF-IDF matrix explicitly for educational transparency and practical lab tasks (Assignment 6).
    """
    cleaned = [d.strip() for d in documents if d.strip()]
    if len(cleaned) < 2:
        return {
            'status': 'insufficient_data',
            'notice': 'Provide at least two documents to compute TF-IDF matrix.'
        }

    doc_word_lists = []
    for doc in cleaned:
        words = [t['text'].casefold() for t in tokenize(doc) if t['is_word']]
        doc_word_lists.append(words)

    # Document frequency count for each term
    term_doc_freq = Counter()
    for words in doc_word_lists:
        unique_in_doc = set(words)
        for term in unique_in_doc:
            term_doc_freq[term] += 1

    n_docs = len(cleaned)
    vocab = [term for term, freq in term_doc_freq.most_common(max_features)]
    if not vocab:
        return {
            'status': 'insufficient_data',
            'notice': 'No valid words found in the provided documents.'
        }

    # Inverse Document Frequency (IDF) with standard smoothed formula
    # idf(t) = ln((1 + N) / (1 + df(t))) + 1.0
    idf_dict = {}
    for term in vocab:
        df = term_doc_freq[term]
        idf_dict[term] = round(math.log((1 + n_docs) / (1 + df)) + 1.0, 4)

    # Term Frequency (TF) and TF-IDF matrix per document
    doc_rows = []
    matrix = []
    for i, words in enumerate(doc_word_lists):
        doc_len = max(len(words), 1)
        term_counts = Counter(words)
        tf_row = {}
        tfidf_row = {}
        raw_vector = []
        for term in vocab:
            count = term_counts.get(term, 0)
            tf_val = round(count / doc_len, 4)
            tf_row[term] = {'count': count, 'tf': tf_val}
            tfidf_val = round(tf_val * idf_dict[term], 4)
            tfidf_row[term] = tfidf_val
            raw_vector.append(tfidf_val)

        # L2 norm for cosine normalization
        norm = math.sqrt(sum(v ** 2 for v in raw_vector))
        normed_vector = [round(v / norm, 4) if norm > 1e-12 else 0.0 for v in raw_vector]
        matrix.append(normed_vector)

        top_terms = sorted(tfidf_row.items(), key=lambda kv: kv[1], reverse=True)[:5]
        doc_rows.append({
            'doc_id': i + 1,
            'text': cleaned[i],
            'word_count': doc_len,
            'tf': tf_row,
            'tfidf': tfidf_row,
            'top_terms': [{'term': t, 'tfidf': s} for t, s in top_terms if s > 0]
        })

    # Pairwise cosine similarity matrix
    sim_matrix = []
    for i in range(n_docs):
        sim_row = []
        for j in range(n_docs):
            if i == j:
                sim_row.append(1.0)
            else:
                sim_val = sum(matrix[i][k] * matrix[j][k] for k in range(len(vocab)))
                sim_row.append(round(sim_val, 4))
        sim_matrix.append(sim_row)

    return {
        'status': 'available',
        'document_count': n_docs,
        'vocabulary': [{'term': t, 'df': term_doc_freq[t], 'idf': idf_dict[t]} for t in vocab],
        'documents': doc_rows,
        'similarity_matrix': sim_matrix,
        'notice': 'TF computed as (term count / document length); smoothed IDF computed as ln((1+N)/(1+DF)) + 1.'
    }


def extractive_qa(context: str, question: str):
    """
    Extractive Question Answering pipeline over input context text (Assignment 9 mini project task).
    Segments context into sentences, ranks candidates by question-to-sentence semantic relevance,
    and returns top candidate answer sentences with score, rationale, and highlighted answer phrase.
    """
    context = context.strip()
    question = question.strip()
    if not context or not question:
        return {
            'status': 'error',
            'notice': 'Please provide both a context passage and a question.'
        }

    cand_sentences = sentences(context)
    if not cand_sentences:
        return {
            'status': 'error',
            'notice': 'Context does not contain any valid sentences.'
        }

    q_tokens = [t['text'].casefold() for t in tokenize(question) if t['is_word']]
    stops = STOPWORDS['en'] | STOPWORDS['hi'] | STOPWORDS['mr']
    q_keywords = [w for w in q_tokens if w not in stops and len(w) > 2]
    if not q_keywords:
        q_keywords = q_tokens

    # TF-IDF sentence similarity
    all_texts = [question] + cand_sentences
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=lambda s: [t['text'].casefold() for t in tokenize(s) if t['is_word']], token_pattern=None)
        matrix = vec.fit_transform(all_texts)
        q_vec = matrix[0]
        s_vecs = matrix[1:]
        sims = cosine_similarity(q_vec, s_vecs).ravel()
    except Exception:
        sims = np.zeros(len(cand_sentences))

    # Identify question intent
    q_lower = question.lower()
    entity_target = None
    if q_lower.startswith(('who', 'whose', 'whom')):
        entity_target = {'PERSON', 'ORGANIZATION'}
    elif q_lower.startswith('where'):
        entity_target = {'GPE', 'LOCATION', 'FACILITY'}
    elif any(k in q_lower for k in ['when', 'year', 'date', 'month', 'time']):
        entity_target = {'DATE', 'TIME'}

    scored_candidates = []
    for idx, (sent, base_sim) in enumerate(zip(cand_sentences, sims)):
        sent_words = set(t['text'].casefold() for t in tokenize(sent) if t['is_word'])
        overlap = [w for w in q_keywords if w in sent_words]
        overlap_score = len(overlap) / max(len(q_keywords), 1)

        # Linguistic named entities
        ling = linguistics(sent, 'en')
        ent_bonus = 0.0
        matched_entities = []
        if ling['status'] == 'available':
            for ent in ling['entities']:
                matched_entities.append(f"{ent['text']} ({ent['type']})")
                if entity_target and ent['type'] in entity_target:
                    ent_bonus += 0.25

        final_score = round(float(0.55 * base_sim + 0.35 * overlap_score + ent_bonus), 4)
        scored_candidates.append({
            'sentence_idx': idx + 1,
            'sentence': sent,
            'score': final_score,
            'matched_keywords': overlap,
            'entities': matched_entities
        })

    scored_candidates.sort(key=lambda c: c['score'], reverse=True)
    best = scored_candidates[0] if scored_candidates else None

    return {
        'status': 'available',
        'question': question,
        'best_answer': best['sentence'] if best and best['score'] > 0.05 else 'No definitive answer found in the provided context.',
        'confidence': best['score'] if best else 0.0,
        'candidates': scored_candidates[:5],
        'total_sentences_evaluated': len(cand_sentences),
        'method': 'Hybrid QA pipeline: TF-IDF vector matching + Keyword Overlap + Named Entity Boosting'
    }
