import asyncio
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend import nlp, news
from scripts.prepare_data import clean_splits, fingerprint

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]


def test_tokens():
    text = 'नमस्कार! विद्यार्थी पुस्तके वाचतात.\nHello, world!'
    tokens = nlp.tokenize(text)
    assert [t['text'] for t in tokens if t['is_word']] == ['नमस्कार', 'विद्यार्थी', 'पुस्तके', 'वाचतात', 'Hello', 'world']
    assert len(tokens) == 10
    assert len(nlp.sentences(text)) == 3
    assert all(text[t['start']:t['end']] == t['text'] for t in tokens)


def test_detection():
    assert nlp.detect('The students are reading books at the library today.')['language'] == 'en'
    assert nlp.detect('पुस्तकालय में छात्र किताबें पढ़ रहे हैं और शिक्षक उनकी मदद कर रहे हैं।')['language'] == 'hi'
    assert nlp.detect('विद्यार्थी वाचनालयात पुस्तके वाचत आहेत आणि शिक्षक त्यांना मदत करत आहेत.')['language'] == 'mr'
    assert nlp.detect('Hello')['uncertain']
    assert nlp.detect('123 !!!')['language'] == 'und'
    assert nlp.detect('आज students library में पढ़ रहे हैं')['uncertain']


def test_english_linguistics():
    text = 'Barack Obama visited New York. Students were reading books.'
    result = nlp.linguistics(text, 'en')
    assert result['status'] == 'available'
    reading = next(r for r in result['rows'] if r['text'] == 'reading')
    assert reading['lemma'] == 'read' and reading['tag'].startswith('V')
    assert result['entities']
    for e in result['entities']:
        assert text[e['start']:e['end']] == e['text']
    assert nlp.linguistics('विद्यार्थी पुस्तके वाचतात.', 'mr')['status'] == 'unsupported'


@pytest.mark.parametrize('lang,text', [('en','The cricket team wins the match in the final tournament'),('hi','क्रिकेट टीम ने मैच जीतकर विश्व कप के फाइनल में प्रवेश किया'),('mr','क्रिकेट संघाने अंतिम सामन्यात विजय मिळवला')])
def test_classifier(lang, text):
    result = nlp.classify(text, lang)
    assert result['category'] in result['labels']
    assert result['features'] and len(result['scores']) == 5
    assert isinstance(result['score'], float)
    assert nlp.classify(text, 'ta')['status'] == 'unavailable'


@pytest.mark.parametrize('lang', ['en', 'hi', 'mr'])
@pytest.mark.parametrize('mode', ['cbow', 'skipgram'])
def test_embeddings(lang, mode):
    result = nlp.embeddings(lang, mode, 'india' if lang == 'en' else 'भारत')
    assert result['status'] == 'available'
    assert len(result['vector']) == 48 and len(result['nearest']) == 8
    assert result['metadata']['training_headlines'] > 10000
    assert nlp.embeddings(lang, mode, 'zzzznotaword')['status'] == 'out_of_vocabulary'


def test_ngram():
    result = nlp.ngram('a b. a c.', 'a b.', n=2, alpha=0)
    assert result['perplexity'] == pytest.approx(2 ** (1 / 3))
    assert nlp.ngram('a b.', 'a z.', 2, 0)['infinite']
    smoothed = nlp.ngram('a b.', 'a z.', 2, 1)
    assert smoothed['oov_count'] == 1 and smoothed['perplexity'] > 0
    assert all(0 < r['probability'] <= 1 for r in smoothed['events'])
    # Evaluation must not alter the training vocabulary.
    assert nlp.ngram('a b.', 'unseen novel tokens', 2, 1)['vocabulary_size'] == 4


def test_lsa():
    result = nlp.lsa(['library books students', 'books reading library', 'cricket team match', 'match team sports'])
    assert result['status'] == 'available'
    assert len(result['coordinates']) == 4
    assert len(result['topics']) <= 3
    assert nlp.lsa(['!!!', '???', '...'])['status'] == 'insufficient_data'


def test_coverage():
    articles = [dict(id='1',title='Students visit the library',description='',language='en'),dict(id='2',title='Students visit the library',description='',language='en'),dict(id='3',title='विद्यार्थी वाचनालयात जातात',description='',language='mr')]
    result = nlp.coverage(articles)
    assert result['matrix'][0][1] == pytest.approx(1)
    assert result['matrix'][0][2] is None
    assert len(result['distinct_terms']) == 3


def test_summary():
    text = 'Students visit a neighbourhood library every afternoon. Volunteers arrange books and prepare a quiet reading room. The community plans to discuss longer opening hours at its next meeting. No final decision has been made about the proposed schedule.'
    result = nlp.summarize(text)
    assert result['status'] == 'available'
    assert all(s in text for s in result['sentences'])
    assert nlp.summarize('Just a headline')['status'] == 'insufficient_text'


def test_deduplication():
    splits = {'train':[{'text':'One useful headline', 'label':'a'},{'text':'Conflict headline','label':'a'}], 'validation':[{'text':'ONE useful headline!', 'label':'a'},{'text':'Conflict headline','label':'b'}], 'test':[{'text':'Different headline', 'label':'a'}]}
    clean, audit = clean_splits(splits)
    assert len(clean['train']) == 1 and not clean['validation']
    assert audit['validation']['removed'] == {'duplicate':1,'conflicting_labels':1}


def test_split_integrity():
    for lang in ('en','hi','mr'):
        files = [ROOT / f'data/prepared/{lang}-{split}.jsonl' for split in ('train','validation','test')]
        if not all(p.exists() for p in files):
            pytest.skip('Raw/prepared training data intentionally excluded from deliverable. Run prepare_data first for leakage audit.')
        sets = [{fingerprint(json.loads(x)['text']) for x in p.read_text(encoding='utf-8').splitlines()} for p in files]
        assert all(not sets[i] & sets[j] for i,j in [(0,1),(0,2),(1,2)])


def test_metrics_integrity():
    import hashlib
    for path in (ROOT/'models').glob('metrics-*.json'):
        data=json.loads(path.read_text())
        assert sum(map(sum,data['confusion_matrix'])) == data['test_count']
        correct=sum(row[i] for i,row in enumerate(data['confusion_matrix']))
        assert correct/data['test_count'] == pytest.approx(data['accuracy'])
        assert data['model_sha256'] == hashlib.sha256((ROOT/f'models/classifier-{data["language"]}.joblib').read_bytes()).hexdigest()
        assert all(data['per_class'][label]['support'] > 0 for label in data['labels'])


def test_provider_normalization():
    rows=[{'title':'One useful headline','url':'https://example.org/news?utm_source=x','seendate':'20261024T120000Z','language':'English'}, {'title':'One useful headline','url':'https://example.org/news?utm_source=y'}, {'title':'Bad','url':'javascript:alert(1)'}, {'title':'Other useful title','url':'https://example.org/other','seendate':'bad'}]
    normalized=news.normalize(rows)
    assert len(normalized)==2
    assert normalized[0]['timestamp_kind']=='first_seen'
    assert normalized[1]['timestamp'] is None
    assert all(x['body']=='' for x in normalized)


def test_offline_filtering():
    results=client.get('/api/news?demo=true&language=mr&topic=education').json()
    assert results['mode']=='demo' and len(results['articles'])==2
    assert all(a['language']=='mr' and a['timestamp'] is None and a['url'] is None for a in results['articles'])
    assert not client.get('/api/news?demo=true&q=zzzzunmatched').json()['articles']
    assert not client.get('/api/news?demo=true&country=US').json()['articles']


def test_provider_outage(monkeypatch):
    class BrokenClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def get(self,*args,**kwargs): raise news.httpx.ConnectError('offline')
    monkeypatch.setattr(news.httpx,'AsyncClient',BrokenClient)
    monkeypatch.setattr(news,'_last_request',0)
    result=asyncio.run(news.get_news(q='uncached-outage-test'))
    assert result['mode']=='unavailable' and result['articles']==[]


def test_provider_query_and_cache(monkeypatch):
    calls=[]
    class GoodClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def get(self,url,params):
            calls.append(params)
            return news.httpx.Response(200,json={'articles':[{'title':'Example headline for query testing','url':'https://example.org/query','language':'English'}]},request=news.httpx.Request('GET',url))
    monkeypatch.setattr(news.httpx,'AsyncClient',GoodClient)
    monkeypatch.setattr(news,'_last_request',0)
    args=dict(q='unique-test-query',language='hi',country='IN',topic='health',sort='relevance')
    first=asyncio.run(news.get_news(**args))
    second=asyncio.run(news.get_news(**args))
    assert first['mode']=='live' and second['cached']
    assert len(calls)==1 and calls[0]['sort']=='HybridRel'
    assert 'sourcelang:hindi' in calls[0]['query'] and 'sourcecountry:IN' in calls[0]['query']


@pytest.mark.parametrize('path,body', [('/api/analyze',{'text':''}),('/api/analyze',{'text':'hello','language':'../bad'}),('/api/embeddings',{'language':'../../bad'}),('/api/ngram',{'training':'a','evaluation':'b','n':8}),('/api/lsa',{'documents':['one']}),('/api/compare',{'articles':[]})])
def test_request_validation(path,body):
    assert client.post(path,json=body).status_code==422


def test_dates_and_language_validation():
    assert client.get('/api/news?start=bad').status_code==422
    assert client.get('/api/news?language=bad').status_code==422
    assert client.get('/api/news?start=2000-01-01&end=2001-01-01').status_code==422


def test_analysis_api():
    result=client.post('/api/analyze',json={'text':'Students are reading books.','language':'en'})
    assert result.status_code==200
    assert result.json()['counts']=={'words':4,'tokens':5,'sentences':1,'characters':27}
    assert 'sentiment' in result.json()
    assert client.post('/api/translate').json()['status']=='unavailable'
    assert len(client.get('/api/models').json()['models'])==3


def test_sentiment_and_emotion():
    res = nlp.sentiment_emotion('The team celebrated their brilliant victory with great joy and happiness!')
    assert res['status'] == 'available'
    assert res['label'] == 'Positive'
    assert res['compound'] > 0.5
    assert res['dominant_emotion'] == 'Joy'
    assert 'joy' in res['evidence']

    neg = nlp.sentiment_emotion('Devastating loss and terrible grief caused deep sorrow and pain.')
    assert neg['label'] == 'Negative'
    assert neg['compound'] < -0.4
    assert neg['dominant_emotion'] == 'Sadness'


def test_opinion_mining():
    opinions = [
        'The library facilities are wonderful and the quiet rooms bring joy to readers.',
        'Volunteers provide great helpful support and readers love this community space.',
        'Traffic congestion near the junction is terrible, causing awful delays and anger.',
        'Broken signals and potholes are deeply frustrating motorists and creating chaos.',
        'The cricket team won the tournament with fantastic teamwork and celebration.'
    ]
    res = client.post('/api/opinion-mining', json={'documents': opinions})
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'available'
    assert data['summary']['total_opinions'] == 5
    assert data['summary']['positive_count'] >= 2
    assert data['summary']['negative_count'] >= 2
    assert len(data['topics']) >= 2


def test_embedding_classification():
    tech_res = client.post('/api/classify-embeddings', json={
        'text': 'Google researchers deploy artificial intelligence software algorithms',
        'language': 'en',
        'mode': 'skipgram'
    })
    assert tech_res.status_code == 200
    data = tech_res.json()
    assert data['status'] == 'available'
    assert data['predicted_category'] == 'technology'
    assert data['similarity_score'] > 0.5
    assert data['words_used']
    assert data['pos_tokens']


def test_vector_analogy_and_keywords():
    analogy_res = client.post('/api/analogy', json={
        'language': 'en',
        'mode': 'skipgram',
        'pos_a': 'bjp',
        'neg_b': 'modi',
        'pos_c': 'rahul'
    })
    assert analogy_res.status_code == 200
    data = analogy_res.json()
    assert data['status'] == 'available'
    assert len(data['results']) > 0

    keywords = nlp.extract_keywords('पुस्तकालय में छात्र पुस्तकें पढ़ रहे हैं और स्वयंसेवक मदद कर रहे हैं।', 'hi')
    assert len(keywords) > 0
    assert any(k['text'] == 'पुस्तकालय' for k in keywords)


def test_web_scraping():
    # Test demo fixture scraping endpoint
    res = client.post('/api/scrape', json={'url': 'https://en.wikipedia.org/wiki/Natural_language_processing'})
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'available'
    assert 'Natural language processing' in data['title']
    assert len(data['paragraphs']) > 0
    assert data['stats']['word_count'] > 30

    # Test invalid url validation
    err_res = client.post('/api/scrape', json={'url': ''})
    assert err_res.status_code == 422


def test_compute_tfidf():
    docs = [
        'Natural language processing involves statistical algorithms and machine learning',
        'Machine learning models learn patterns from text data and language processing',
        'Cricket match was exciting with team victory in tournament'
    ]
    res = client.post('/api/tfidf', json={'documents': docs})
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'available'
    assert data['document_count'] == 3
    assert len(data['vocabulary']) > 0
    assert len(data['documents']) == 3
    assert len(data['similarity_matrix']) == 3
    # Check that doc 1 and doc 2 have higher cosine similarity than doc 1 and doc 3
    assert data['similarity_matrix'][0][1] > data['similarity_matrix'][0][2]


def test_ngram_sequence_probability():
    training = 'the library welcomes readers. students read books in the library.'
    evaluation = 'students read books.'
    res = nlp.ngram(training, evaluation, n=2, alpha=1.0)
    assert res['sequence_probability'] > 0
    assert res['log_probability'] is not None
    assert res['perplexity'] > 0


def test_extractive_qa():
    context = (
        'The Indian Space Research Organisation (ISRO) successfully launched the Chandrayaan-3 mission from Sriharikota. '
        'Dr. Sreedhara Panicker Somanath served as the chairman during this historic lunar mission. '
        'The spacecraft landed near the south pole of the Moon on August 23, 2023.'
    )
    res = client.post('/api/qa', json={
        'context': context,
        'question': 'Who served as the chairman during the lunar mission?'
    })
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'available'
    assert 'Somanath' in data['best_answer']
    assert data['confidence'] > 0
    assert len(data['candidates']) > 0


