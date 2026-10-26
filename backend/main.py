import json
from datetime import date, timedelta
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from . import nlp, news, scraper

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='SamacharSetu local API', version='1.0.0')


class ScrapeRequest(BaseModel):
    url: str = Field(min_length=3, max_length=1000)


class TfidfRequest(BaseModel):
    documents: list[str] = Field(min_length=2, max_length=15)


class QaRequest(BaseModel):
    context: str = Field(min_length=10, max_length=20000)
    question: str = Field(min_length=3, max_length=500)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    language: str = 'auto'


class EmbeddingRequest(BaseModel):
    language: Literal['en', 'hi', 'mr'] = 'en'
    mode: Literal['cbow', 'skipgram'] = 'skipgram'
    word: str = Field(default='', max_length=80)
    other: str = Field(default='', max_length=80)


class AnalogyRequest(BaseModel):
    language: Literal['en', 'hi', 'mr'] = 'en'
    mode: Literal['cbow', 'skipgram'] = 'skipgram'
    pos_a: str = Field(min_length=1, max_length=80)
    neg_b: str = Field(min_length=1, max_length=80)
    pos_c: str = Field(min_length=1, max_length=80)


class Article(BaseModel):
    id: str = Field(max_length=100)
    title: str = Field(min_length=1, max_length=1000)
    description: str = Field(default='', max_length=4000)
    language: str = Field(max_length=10)


class CompareRequest(BaseModel):
    articles: list[Article] = Field(min_length=2, max_length=6)


class NgramRequest(BaseModel):
    training: str = Field(min_length=1, max_length=20000)
    evaluation: str = Field(min_length=1, max_length=6000)
    n: int = Field(default=2, ge=1, le=3)
    alpha: float = Field(default=1., ge=0., le=5.)


class LsaRequest(BaseModel):
    documents: list[str] = Field(min_length=3, max_length=30)


class OpinionMiningRequest(BaseModel):
    documents: list[str] = Field(min_length=3, max_length=30)


class EmbeddingClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    language: Literal['en', 'hi', 'mr'] = 'en'
    mode: Literal['cbow', 'skipgram'] = 'skipgram'



@app.get('/api/health')
def health():
    return {'status': 'ok', 'app': 'SamacharSetu'}


@app.get('/api/models')
def models():
    return {'models': [json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT / 'models').glob('metrics-*.json'))],
            'languages': nlp.LANGUAGES, 'coverage': {'tokenization': list(nlp.LANGUAGES), 'classification': [x for x in ('en', 'hi', 'mr') if (ROOT / f'models/classifier-{x}.joblib').exists()], 'pos_ner_normalization': ['en'], 'translation': [], 'interface': ['en', 'hi', 'mr'], 'cross_language_similarity': False}}


@app.get('/api/news')
async def get_news(q: str = Query('', max_length=160), language: str = 'all', country: str = '', topic: str = 'all', sort: Literal['newest', 'relevance'] = 'newest', start: str = '', end: str = '', demo: bool = False):
    if language != 'all' and language not in news.LANGS:
        raise HTTPException(422, 'Unsupported news language')
    if country and country not in news.COUNTRIES:
        raise HTTPException(422, 'Unsupported source country')
    if topic != 'all' and topic not in news.TOPIC_QUERY:
        raise HTTPException(422, 'Unsupported topic')
    dates = {}
    for key, value in [('start', start), ('end', end)]:
        if value:
            try:
                dates[key] = date.fromisoformat(value)
            except ValueError:
                raise HTTPException(422, 'Use ISO dates (YYYY-MM-DD).')
            if dates[key] > date.today() or dates[key] < date.today() - timedelta(days=90):
                raise HTTPException(422, 'GDELT search is limited to the last 90 days.')
    if len(dates) == 2 and dates['start'] > dates['end']:
        raise HTTPException(422, 'Start must be before end.')
    if bool(start) != bool(end):
        raise HTTPException(422, 'Choose both start and end dates.')
    return await news.get_news(q, language, country, topic, sort, start, end, demo)


@app.post('/api/analyze')
def analyze(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(422, 'Enter some text.')
    if request.language != 'auto' and request.language not in nlp.LANGUAGES:
        raise HTTPException(422, 'Unsupported language override')
    return nlp.analyze(request.text, request.language)


@app.post('/api/embeddings')
def embeddings(request: EmbeddingRequest):
    return nlp.embeddings(request.language, request.mode, request.word, request.other)


@app.post('/api/analogy')
def analogy(request: AnalogyRequest):
    return nlp.vector_analogy(request.language, request.mode, request.pos_a, request.neg_b, request.pos_c)


@app.post('/api/compare')
def compare(request: CompareRequest):
    return nlp.coverage([a.model_dump() for a in request.articles])


@app.post('/api/ngram')
def ngram(request: NgramRequest):
    if not any(t['is_word'] for t in nlp.tokenize(request.training)) or not any(t['is_word'] for t in nlp.tokenize(request.evaluation)):
        raise HTTPException(422, 'Training and evaluation need words.')
    return nlp.ngram(request.training, request.evaluation, request.n, request.alpha)


@app.post('/api/lsa')
def lsa(request: LsaRequest):
    if sum(map(len, request.documents)) > 20000:
        raise HTTPException(422, 'Use at most 20,000 characters across documents.')
    return nlp.lsa(request.documents)


@app.post('/api/opinion-mining')
def opinion_mining(request: OpinionMiningRequest):
    if sum(map(len, request.documents)) > 20000:
        raise HTTPException(422, 'Use at most 20,000 characters across documents.')
    return nlp.opinion_mining(request.documents)


@app.post('/api/classify-embeddings')
def classify_embeddings(request: EmbeddingClassifyRequest):
    return nlp.classify_with_embeddings(request.text, request.language, request.mode)


@app.post('/api/sentiment')
def sentiment(request: TextRequest):
    return nlp.sentiment_emotion(request.text, request.language)


@app.post('/api/translate')
def translate():
    return {'status': 'unavailable', 'notice': 'No translation provider is configured. The original text is retained; your text has not been sent elsewhere.'}


@app.post('/api/scrape')
async def scrape_endpoint(request: ScrapeRequest):
    return await scraper.scrape_web_data(request.url)


@app.post('/api/tfidf')
def tfidf_endpoint(request: TfidfRequest):
    if sum(map(len, request.documents)) > 20000:
        raise HTTPException(422, 'Use at most 20,000 characters across documents.')
    return nlp.compute_tfidf(request.documents)


@app.post('/api/qa')
def qa_endpoint(request: QaRequest):
    return nlp.extractive_qa(request.context, request.question)

