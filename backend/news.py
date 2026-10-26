"""Bounded server-side GDELT adapter. Never downloads publisher article bodies."""
import asyncio
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import httpx
from .nlp import classify

ROOT = Path(__file__).resolve().parents[1]
LANGS = {'en': 'english', 'hi': 'hindi', 'mr': 'marathi', 'bn': 'bengali', 'ta': 'tamil', 'te': 'telugu', 'gu': 'gujarati', 'kn': 'kannada', 'ml': 'malayalam', 'pa': 'punjabi', 'or': 'oriya', 'as': 'assamese'}
COUNTRIES = {'IN': 'IN', 'US': 'US', 'GB': 'UK'}
TOPIC_QUERY = {'civic': '(water OR roads OR sanitation)', 'politics': '(election OR parliament OR government)', 'business': '(economy OR business)', 'education': '(education OR university)', 'health': '(health OR hospital)', 'technology': '(science OR technology)', 'environment': '(climate OR environment)', 'sports': '(sports OR cricket)', 'entertainment': '(cinema OR entertainment)'}
_cache = OrderedDict()
_lock = asyncio.Lock()
_last_request = 0.


def safe_url(url):
    try:
        parsed = urlsplit(url or '')
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            return None
        return url
    except ValueError:
        return None


def canonical_url(url):
    p = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith('utm_') and k not in ('fbclid', 'gclid')]
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip('/'), urlencode(query), ''))


def normalize(items):
    seen_urls, seen_titles, results = set(), set(), []
    reverse = {v: k for k, v in LANGS.items()}
    for raw in items:
        url = safe_url(raw.get('url'))
        title = (raw.get('title') or '').strip()
        if not url or not title:
            continue
        canonical, key = canonical_url(url), re.sub(r'\W+', '', title.casefold())
        if canonical in seen_urls or key in seen_titles:
            continue
        seen_urls.add(canonical)
        seen_titles.add(key)
        lang = reverse.get((raw.get('language') or '').casefold(), 'und')
        date = None
        try:
            date = datetime.strptime(raw.get('seendate', ''), '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
        results.append({'id': hashlib.sha256(canonical.encode()).hexdigest()[:16], 'title': title,
                        'url': url, 'publisher': urlsplit(url).hostname, 'language': lang,
                        'timestamp': date, 'timestamp_kind': 'first_seen', 'description': '', 'body': '',
                        'source_country': raw.get('sourcecountry') or '',
                        'image': None, 'mode': 'live', 'prediction': classify(title, lang)})
    return results


def fixture_results(q='', language='all', topic='all', country='', start='', end='', **kwargs):
    articles = json.loads((ROOT / 'data/demo/stories.json').read_text(encoding='utf-8'))
    terms = q.casefold().split()
    results = [a for a in articles if (language == 'all' or a['language'] == language)
               and (topic == 'all' or a['topic'] == topic)
               and (not country or country == 'IN')
               and all(t in (a['title'] + a['description']).casefold() for t in terms)]
    # Fictional fixtures have no publication date: do not invent date-filter matches.
    if start or end:
        results = []
    for article in results:
        article['prediction'] = classify(article['title'], article['language'])
    return results


async def get_news(q='', language='all', country='', topic='all', sort='newest', start='', end='', demo=False):
    global _last_request
    args = dict(q=q, language=language, country=country, topic=topic, sort=sort, start=start, end=end)
    if demo:
        return {'mode': 'demo', 'articles': fixture_results(**args), 'notice': 'Offline learning fixtures. Fictional scenarios written for this project; these are not reported events or current news.', 'cached': False}
    key = tuple(args.values())
    now = time.monotonic()
    ttl = int(os.getenv('GDELT_CACHE_SECONDS', '300'))
    if key in _cache and now - _cache[key][0] < ttl:
        return {**_cache[key][1], 'cached': True}
    terms = re.findall(r'[^\W_]+', q, flags=re.UNICODE)[:12]
    query = ' '.join('"' + word + '"' for word in terms) or 'India'
    if language != 'all':
        query += ' sourcelang:' + LANGS[language]
    if country:
        query += ' sourcecountry:' + COUNTRIES[country]
    if topic != 'all':
        query += ' ' + TOPIC_QUERY[topic]
    params = {'query': query, 'mode': 'ArtList', 'format': 'json', 'maxrecords': 50,
              'sort': 'DateDesc' if sort == 'newest' else 'HybridRel'}
    if start:
        params['startdatetime'] = start.replace('-', '') + '000000'
    if end:
        params['enddatetime'] = end.replace('-', '') + '235959'
    if not start and not end:
        params['timespan'] = '24h'
    try:
        async with _lock:
            # Avoid parallel provider calls and repeated fetching on every keystroke.
            if key in _cache and time.monotonic() - _cache[key][0] < ttl:
                return {**_cache[key][1], 'cached': True}
            wait = 5 - (time.monotonic() - _last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            _last_request = time.monotonic()
            async with httpx.AsyncClient(timeout=float(os.getenv('GDELT_TIMEOUT_SECONDS', '15')), follow_redirects=False) as client:
                response = await client.get('https://api.gdeltproject.org/api/v2/doc/doc', params=params)
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get('articles'), list):
                raise ValueError('Unexpected provider response')
            result = {'mode': 'live', 'articles': normalize(payload['articles']), 'cached': False,
                      'fetched_at': datetime.now(timezone.utc).isoformat(),
                      'notice': 'GDELT metadata · up to 50 matches. Times are first observed by GDELT, not verified publication times. Topic filters use search terms, not classifier labels.'}
            _cache[key] = (time.monotonic(), result)
            while len(_cache) > 64:
                _cache.popitem(last=False)
            return result
    except (httpx.HTTPError, ValueError, TypeError):
        return {'mode': 'unavailable', 'articles': [], 'cached': False, 'notice': 'Live news could not be reached. Retry later or choose the explicitly labelled offline examples.'}
