'use client';
import {useEffect,useState} from 'react';
import {ArrowRight,FlaskConical,Play,ChevronRight,CheckCircle2,BookOpen,Globe,Search,HelpCircle} from 'lucide-react';
import {api,languages,pct} from '@/lib/api';

const tabs=['Text analysis','Web scraper','Word embeddings','Advanced NLP','Opinion & sentiment','QA & Pipeline','Model evaluation','Assignment map'];
const sample='Students in Pune are reading books at a community library. Microsoft and Google develop language technology. The volunteers discussed accessible entrances and longer reading hours.';
const mapping=[
 ['A','Language detection','Text analysis → Language','langid.py pretrained classifier','English + Indic languages; short/mixed text flagged','test_detection'],
 ['B','Tokenization & statistics','Text analysis → Tokens','Unicode mark-aware regex + sentence boundary rules','All listed languages; not a learned sentence segmenter','test_tokens'],
 ['C','Stemming & lemmatization','Text analysis → Normalization','Snowball + POS-aware WordNet','English','test_english_linguistics'],
 ['D','Part-of-speech tagging','Text analysis → POS','NLTK averaged perceptron','English','test_english_linguistics'],
 ['E','Named entities','Text analysis → Entities','NLTK maximum-entropy chunker','English; predicted spans','test_english_linguistics'],
 ['F','Word vectors & similarity','Word embeddings','Gensim Word2Vec + cosine','English, Hindi, Marathi','test_embeddings'],
 ['G','News classification','Text analysis → Category','Word + character TF-IDF / LinearSVC','Language-specific English, Hindi, Marathi labels','test_classifier'],
 ['H','N-grams & smoothing','Advanced NLP → Language model','Unigram/bigram/trigram, MLE or add-α','Unicode words; user-supplied corpus','test_ngram'],
 ['I','Perplexity','Advanced NLP → Language model','exp(−mean log probability) on separate evaluation text','Same vocabulary and evaluation text for comparison','test_ngram'],
 ['J','CBOW & Skip-gram','Word embeddings → Training objective','Two separately trained Word2Vec models','English, Hindi, Marathi','test_embeddings'],
 ['K','Vector semantics & LSA','Advanced NLP → Topic exploration','TF-IDF, truncated SVD, component inspection','Unicode text; best within one language','test_lsa'],
 ['L','Related coverage','Story reader + Compare','Same-language character TF-IDF / cosine','Cross-language comparison explicitly unavailable','test_coverage']
];
const syllabusMapping=[
 ['1','Web Scraping to Gather Data','Web scraper tab / scripts/assignment1_web_scraping.py','HTTP client + BeautifulSoup HTML extraction (title, headings, paragraphs, links)','Web URLs & offline fixtures','test_web_scraping'],
 ['2','Language Detection, Word & Sentence Count, Tokenization','Text analysis → Tokens & Stats / scripts/assignment2_language_detection_tokenization.py','Pretrained langid.py + Unicode mark-aware regex & boundary rules','English, Hindi, Marathi + 9 Indian languages','test_tokens, test_detection'],
 ['3','Stemming & WordNet Lemmatization','Text analysis → Normalization / scripts/assignment3_stemming_lemmatization.py','Snowball stemmer + POS-aware WordNet lemmatizer','English (with Indic explanations)','test_english_linguistics'],
 ['4','POS Tagging, Word Embeddings & Text Classification','Word embeddings → Classifier / scripts/assignment4_pos_embedding_classification.py','NLTK averaged perceptron + Word2Vec 48-dim centroid cosine classification','English, Hindi, Marathi','test_embedding_classification'],
 ['5','Named Entity Recognition (NER)','Text analysis → Named entities / scripts/assignment5_named_entity_recognition.py','NLTK Maximum-Entropy Entity Chunker','English (highlighted spans & entity types)','test_english_linguistics'],
 ['6','TF-IDF Method (Term Frequency - Inverse Document Frequency)','Advanced NLP → TF-IDF explorer / scripts/assignment6_tfidf.py','Term Frequency TF(t,d) * Smoothed IDF ln((1+N)/(1+DF))+1 & Cosine Matrix','Multilingual document collections','test_compute_tfidf'],
 ['7','N-gram Models & Sequence Probability Determination','Advanced NLP → Language model / scripts/assignment7_ngram_language_model.py','Unigram, Bigram, Trigram MLE + Laplace Add-α smoothing, Sequence P(W) & Perplexity','Unicode word sequences','test_ngram_sequence_probability'],
 ['8','Latent Semantic Indexing (LSI) for Categorization & Opinion Mining','Advanced NLP → LSA / Opinion & sentiment / scripts/assignment8_latent_semantic_indexing.py','TF-IDF + TruncatedSVD dimensionality reduction & VADER emotion stance clustering','Document collections & public feedback','test_lsa, test_opinion_mining'],
 ['9','Mini Project: End-to-End NLP Pipeline with QA System','QA & Pipeline tab / scripts/assignment9_nlp_pipeline_miniproject.py','Complete pipeline: Scraping -> Preprocessing -> Linguistics -> Embeddings -> Sentiment -> Extractive QA','Full news articles & question answering','test_extractive_qa']
];

export default function Lab(){
  const [tab,setTab]=useState('Text analysis'),[text,setText]=useState(sample),[language,setLanguage]=useState('auto'),[result,setResult]=useState<any>(null),[busy,setBusy]=useState(false),[error,setError]=useState(''),[models,setModels]=useState<any>(null),[modelsError,setModelsError]=useState('');
  useEffect(()=>{try{const input=JSON.parse(sessionStorage.getItem('ss-lab-input')||'null');if(input){setText(input.text);setLanguage(input.language in languages?input.language:'auto');sessionStorage.removeItem('ss-lab-input');}}catch{}api('models').then(setModels).catch(e=>setModelsError(e.message));},[]);
  async function run(){setBusy(true);setError('');setResult(null);try{setResult(await api('analyze',{text,language}));}catch(e:any){setError(e.message);}finally{setBusy(false);}}
  return <><div className="lab-heading"><div><div className="eyebrow">THE LANGUAGE BEHIND THE NEWS</div><h1>NLP Learning Lab<span className="lab-period">.</span></h1><p>Bring a headline. Follow the steps. See what the algorithms actually do.</p></div><div className="local-badge"><CheckCircle2 size={17}/> Your text stays on this machine</div></div><div className="lab-tabs" role="tablist" aria-label="Learning modules">{tabs.map(t=><button role="tab" aria-selected={tab===t} aria-controls="lab-panel" id={'tab-'+t.replaceAll(' ','-')} className={tab===t?'selected':''} key={t} onClick={()=>setTab(t)}>{t}</button>)}</div><div id="lab-panel" role="tabpanel" aria-labelledby={'tab-'+tab.replaceAll(' ','-')}>
  {tab==='Text analysis'&&<><div className="lab-input"><div className="section-heading"><h2>Your source text</h2><label>Analysis language <select aria-label="Analysis language" value={language} onChange={e=>{setLanguage(e.target.value);setResult(null);}}><option value="auto">Detect automatically</option>{Object.entries(languages).map(([k,v])=><option value={k} key={k}>{v}</option>)}</select></label></div><div className="sample-pills" aria-label="Quick test samples"><span className="sample-label">Load sample:</span><button type="button" className="sample-pill" onClick={()=>{setText(sample);setLanguage('en');setResult(null);}}>English</button><button type="button" className="sample-pill" onClick={()=>{setText('पुणे के छात्र एक सामुदायिक पुस्तकालय में पुस्तकें पढ़ रहे हैं। स्वयंसेवकों ने अध्ययन कक्ष के समय को बढ़ाने और प्रवेश को सुलभ बनाने पर चर्चा की।');setLanguage('hi');setResult(null);}}>हिन्दी</button><button type="button" className="sample-pill" onClick={()=>{setText('पुण्यातील विद्यार्थी एका सार्वजनिक वाचनालयात पुस्तके वाचत आहेत. स्वयंसेवकांनी वाचनालयाची वेळ वाढवण्यावर आणि प्रवेश अधिक सुलभ करण्यावर चर्चा केली.');setLanguage('mr');setResult(null);}}>मराठी</button></div><label className="sr-only" htmlFor="source-text">Text to analyse</label><textarea id="source-text" value={text} maxLength={12000} onChange={e=>{setText(e.target.value);setResult(null);}} rows={5}/><div className="input-footer"><p>{text.length.toLocaleString()} / 12,000 characters · Paste only text you want to analyse.</p><button className="primary" disabled={busy||!text.trim()} onClick={run}><Play size={15}/>{busy?'Analysing…':'Analyse text'}</button></div></div>{error&&<p className="notice error" role="alert">{error}</p>}{!result&&!busy&&<div className="lab-welcome"><FlaskConical size={28}/><div><h3>A transparent view of the pipeline.</h3><p>Run the analysis to inspect language, tokens, word normalization, POS, entities and the trained category classifier. Missing capabilities are explained in place.</p></div></div>}{busy&&<p role="status" className="loading">Running local NLP models…</p>}{result&&<><div className="statistics">{Object.entries(result.counts).map(([k,v])=><div key={k}><strong>{Number(v).toLocaleString()}</strong><span>{k}</span></div>)}</div><div className="analysis-grid"><Module letter="A" title="Language detection" subtitle="Identify the language before selecting a pipeline." open><p className="prediction">{languages[result.detection.language]||result.detection.language}<span className="pill">{result.detection.uncertain?'Check language':'Detected'}</span></p><p>{result.detection.confidence!==null?'Model confidence: '+pct(result.detection.confidence):'No confidence available'}</p><p className="fineprint">{result.detection.method}. {result.detection.notice}</p><p className="fineprint">Analysis pipeline: <strong>{languages[result.language]||result.language}</strong>{language!=='auto'?' (selected by you)':''}.</p></Module><Module letter="G" title="News category" subtitle="A trained classifier, with inspectable evidence." open>{result.classification.category?<><p className="prediction">{result.classification.category}</p><p>SVM score: <strong>{result.classification.score}</strong> · {result.classification.status==='uncertain'?'Uncertain prediction':'Model suggestion'}</p><p className="fineprint">{result.classification.notice}</p><details><summary>Supported labels & feature contributions</summary><p>{result.classification.labels.join(' · ')}</p><div className="table-scroll"><table><thead><tr><th>Feature</th><th>Contribution</th></tr></thead><tbody>{result.classification.features.map((f:any)=><tr key={f.feature}><td>{f.feature}</td><td>{f.contribution}</td></tr>)}</tbody></table></div><p className="fineprint">Positive linear contributions to the selected class; these associations are not a causal explanation.</p></details></>:<p className="muted">{result.classification.notice}</p>}</Module></div>
  <Module letter="B" title="Tokens & text statistics" subtitle="A word, a token and a sentence are different units." open><p className="fineprint">Words contain letters or digits. Token count also includes punctuation. Combining marks stay with Indic words. Sentences split at line breaks or sentence punctuation followed by whitespace; abbreviations may be split incorrectly.</p><div className="tokens token-stream">{result.tokens.map((t:any,i:number)=><span className={!t.is_word?'punctuation':''} key={i} title={`Character span ${t.start}–${t.end}`}>{t.text}</span>)}</div><details><summary>{result.sentences.length} sentence segments</summary><ol>{result.sentences.map((s:string,i:number)=><li key={i}>{s}</li>)}</ol></details></Module>
  <Module letter="C / D" title="Normalization & part-of-speech" subtitle="Compare stems, dictionary lemmas and contextual tags." open><p className="fineprint">Stemming trims word endings; lemmatization uses a dictionary form and the predicted POS. Example: “studies” → stem “studi”, lemma “study”. {result.linguistics.notice}</p>{result.linguistics.rows.length>0&&<div className="table-scroll"><table><caption>English token analysis</caption><thead><tr><th>Token</th><th>Stem</th><th>Lemma</th><th>POS</th><th>Readable tag</th></tr></thead><tbody>{result.linguistics.rows.map((r:any,i:number)=><tr key={i}><td>{r.text}</td><td>{r.stem}</td><td>{r.lemma}</td><td><code>{r.tag}</code></td><td>{r.description}</td></tr>)}</tbody></table></div>}<p className="fineprint">POS tags describe grammatical roles in this sentence. A noun can name a person or place; a verb describes an action or state. Statistical predictions can make mistakes.</p></Module>
  <Module letter="E" title="Named entities" subtitle="Inspect exactly which spans the model identified." open>{result.linguistics.entities.length?<><EntityText text={text} entities={result.linguistics.entities}/><div className="table-scroll"><table><thead><tr><th>Span</th><th>Type</th><th>Character range</th></tr></thead><tbody>{result.linguistics.entities.map((e:any,i:number)=><tr key={i}><td>{e.text}</td><td>{e.type}</td><td>{e.start}–{e.end}</td></tr>)}</tbody></table></div></>:<p className="muted">{result.linguistics.status==='available'?'No supported entities were detected. This is a valid result; entities are not invented.':result.linguistics.notice}</p>}</Module>{result.sentiment&&result.sentiment.status==='available'&&<Module letter="10" title="Sentiment & emotion" subtitle="Polarity score and basic emotion detection." open><div className="analysis-grid"><div className="paper-panel"><p className="prediction">{result.sentiment.label}<small>Compound polarity: {result.sentiment.compound>0?'+':''}{result.sentiment.compound.toFixed(4)}</small></p><p className="fineprint">{result.sentiment.notice}</p><div className="similarity-row"><span>Positive</span><div className="bar-track"><span style={{width:Math.max(0,result.sentiment.polarity.pos)*100+'%'}}/></div><code>{(result.sentiment.polarity.pos*100).toFixed(1)}%</code></div><div className="similarity-row"><span>Neutral</span><div className="bar-track"><span style={{width:Math.max(0,result.sentiment.polarity.neu)*100+'%'}}/></div><code>{(result.sentiment.polarity.neu*100).toFixed(1)}%</code></div><div className="similarity-row"><span>Negative</span><div className="bar-track"><span style={{width:Math.max(0,result.sentiment.polarity.neg)*100+'%'}}/></div><code>{(result.sentiment.polarity.neg*100).toFixed(1)}%</code></div></div><div className="paper-panel"><p className="prediction">{result.sentiment.dominant_emotion}<small>Dominant emotion</small></p>{Object.entries(result.sentiment.emotions).map(([emo,val]:[string,any])=><div className="similarity-row" key={emo}><span style={{textTransform:'capitalize'}}>{emo}</span><div className="bar-track"><span style={{width:Math.max(0,Number(val))*100+'%'}}/></div><code>{(Number(val)*100).toFixed(1)}%</code></div>)}</div></div></Module>}</> }</>}
  {tab==='Web scraper'&&<WebScraperLab onSendToAnalysis={(scrapedText:string)=>{setText(scrapedText);setTab('Text analysis');setResult(null);}}/>}
  {tab==='Word embeddings'&&<EmbeddingLab/>}
  {tab==='Advanced NLP'&&<AdvancedLab/>}
  {tab==='Opinion & sentiment'&&<OpinionLab/>}
  {tab==='QA & Pipeline'&&<QaPipelineLab/>}
  {tab==='Model evaluation'&&(models?<Evaluation data={models}/>:<p className="notice">{modelsError||'Loading measured evaluation results…'}</p>)}
  {tab==='Assignment map'&&<><div className="section-heading"><div><h2>Every concept has a working place.</h2><p>Official university practical curriculum (Assignments 1–9) and conceptual modules mapped to code and tests.</p></div><BookOpen size={30}/></div><div className="section-heading" style={{margin:'14px 0 10px'}}><div><h3>University Practical Syllabus (Assignments 1–9)</h3><p>Direct 1-to-1 mapping of the required lab curriculum.</p></div></div><div className="table-scroll"><table className="extended-mapping-table"><thead><tr><th>#</th><th>Academic task & location</th><th>Algorithm</th><th>Coverage & test</th></tr></thead><tbody>{syllabusMapping.map(([id,name,location,algorithm,coverage,test])=><tr key={id}><td><span className="module-letter" style={{background:'var(--accent)',color:'#fff'}}>{id}</span></td><td><strong>{name}</strong><small>{location}</small></td><td>{algorithm}</td><td>{coverage}<small><code>{test}</code></small></td></tr>)}</tbody></table></div><div className="section-heading" style={{margin:'28px 0 14px'}}><div><h3>Detailed Conceptual Modules (A–L)</h3><p>Granular architectural components across the platform.</p></div></div><div className="table-scroll"><table className="mapping-table"><thead><tr><th>ID</th><th>Academic task & location</th><th>Algorithm</th><th>Coverage & test</th></tr></thead><tbody>{mapping.map(([id,name,location,algorithm,coverage,test])=><tr key={id}><td><span className="module-letter">{id}</span></td><td><strong>{name}</strong><small>{location}</small></td><td>{algorithm}</td><td>{coverage}<small><code>{test}</code></small></td></tr>)}</tbody></table></div><p className="fineprint">The full function-level mapping is in docs/assignment-mapping.md. Optional translation, neural language models and LDA are not claimed as implemented assignments.</p></>}
  </div></>;
}

function Module({letter,title,subtitle,children,open=false}:{letter:string;title:string;subtitle:string;children:React.ReactNode;open?:boolean}){return <details className="module" open={open}><summary><span className="module-letter">{letter}</span><div><h2>{title}</h2><p>{subtitle}</p></div><ChevronRight size={18}/></summary><div className="module-body">{children}</div></details>;}
function EntityText({text,entities}:{text:string;entities:any[]}){const parts:React.ReactNode[]=[];let last=0;const chars=Array.from(text);entities.forEach((e,i)=>{parts.push(chars.slice(last,e.start).join(''));parts.push(<mark key={i} title={e.type}>{e.text}<small>{e.type}</small></mark>);last=e.end;});parts.push(chars.slice(last).join(''));return <p className="entity-text">{parts}</p>;}

function EmbeddingLab(){const [language,setLanguage]=useState('en'),[mode,setMode]=useState('skipgram'),[word,setWord]=useState('india'),[other,setOther]=useState('government'),[result,setResult]=useState<any>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false);async function run(){setBusy(true);setError('');try{setResult(await api('embeddings',{language,mode,word,other}));}catch(e:any){setError(e.message);}finally{setBusy(false);}}return <><div className="concept-pair"><div><span className="eyebrow">CBOW</span><h2>Context → word</h2><p>Predict a missing word from its neighbouring words. Here, each training headline supplies a context window of four.</p></div><div><span className="eyebrow">SKIP-GRAM</span><h2>Word → context</h2><p>Predict neighbouring words from a centre word. These are separately trained models, not two names for the same vectors.</p></div></div><div className="lab-input"><div className="filter-fields"><label>Language<select value={language} onChange={e=>{setLanguage(e.target.value);setWord(e.target.value==='hi'?'भारत':e.target.value==='mr'?'भारत':'india');setOther('');setResult(null);}}><option value="en">English</option><option value="hi">हिन्दी</option><option value="mr">मराठी</option></select></label><label>Training objective<select value={mode} onChange={e=>{setMode(e.target.value);setResult(null);}}><option value="skipgram">Skip-gram</option><option value="cbow">CBOW</option></select></label><label>Word<input aria-label="Embedding word" value={word} onChange={e=>setWord(e.target.value)} maxLength={80}/></label><label>Compare with<input value={other} onChange={e=>setOther(e.target.value)} maxLength={80}/></label><button className="primary" onClick={run} disabled={busy}>{busy?'Looking up…':'Inspect vectors'}<ArrowRight size={16}/></button></div></div>{error&&<p className="notice error" role="alert">{error}</p>}{result&&<><p className="notice">{result.metadata?`${result.metadata.algorithm} · ${result.metadata.training_headlines.toLocaleString()} training headlines · ${result.metadata.vocabulary.toLocaleString()} vocabulary words · ${result.metadata.dimensions} dimensions` : result.notice}</p>{result.status==='available'?<div className="analysis-grid"><div className="paper-panel"><h2>Neighbours of “{result.word}”</h2><p className="fineprint">Cosine similarity: association learned from the corpus, not dictionary equivalence.</p>{result.nearest.map((r:any)=><div className="similarity-row" key={r.word}><span>{r.word}</span><div className="bar-track"><span style={{width:Math.max(0,r.cosine)*100+'%'}}/></div><code>{r.cosine.toFixed(3)}</code></div>)}</div><div className="paper-panel"><h2>Vector inspection</h2>{result.pair_similarity!==null?<p className="prediction">{result.pair_similarity.toFixed(4)}<small>cosine similarity to the second word</small></p>:<p>{result.pair_notice||'Enter another known word to compare.'}</p>}<details><summary>All {result.vector.length} learned dimensions</summary><div className="vector-values">{result.vector.map((n:number,i:number)=><span key={i}><small>{i+1}</small>{n.toFixed(5)}</span>)}</div></details><p className="fineprint">Word vectors are trained on the clean training split only. They are not multilingual sentence embeddings; cross-language vector comparisons are not supported. Neighbour quality has not been independently benchmarked.</p></div></div>:<div className="empty"><p>{result.notice}</p>{result.examples&&<div className="tokens">{result.examples.map((w:string)=><button key={w} onClick={()=>setWord(w)}>{w}</button>)}</div>}</div>}</>}<EmbeddingClassifierSection language={language} mode={mode}/></>}

function AdvancedLab(){const [training,setTraining]=useState('the library opens early.\nthe library welcomes readers.\nstudents read books.\nreaders enjoy the library.\nthe community supports students.\nstudents visit the library.'),[evaluation,setEvaluation]=useState('students read books.'),[n,setN]=useState(2),[alpha,setAlpha]=useState(1),[ng,setNg]=useState<any>(null),[documents,setDocuments]=useState('Students visit the library and borrow books.\nA library reading room helps students study.\nThe cricket team practises on the sports ground.\nPlayers prepare for a cricket match.\nRainwater collection can reduce water waste.\nResidents repair leaking taps to save water.'),[ls,setLs]=useState<any>(null),[error,setError]=useState(''),[busy,setBusy]=useState('');async function run(which:string){setError('');setBusy(which);try{if(which==='ng')setNg(await api('ngram',{training,evaluation,n,alpha}));else setLs(await api('lsa',{documents:documents.split('\n').map(x=>x.trim()).filter(Boolean)}));}catch(e:any){setError(e.message);}finally{setBusy('');}}return <>{error&&<p className="notice error" role="alert">{error}</p>}<Module letter="H / I" title="Language modelling & sequence probability" subtitle="Fit counts on one text. Evaluate word sequence probabilities and perplexity." open><div className="formula">P(w | h) = (count(h,w) + α) / (count(h) + α|V|)<br/><span>P(W) = ∏ P(wᵢ | hᵢ) · Perplexity = exp(− Σ log P(wᵢ | hᵢ) / N)</span></div><div className="analysis-grid"><label>Training corpus<textarea rows={7} value={training} onChange={e=>setTraining(e.target.value)} maxLength={20000}/></label><div><label>Evaluation text sequence<textarea rows={3} value={evaluation} onChange={e=>setEvaluation(e.target.value)} maxLength={6000}/></label><div className="filter-fields"><label>Order<select value={n} onChange={e=>setN(Number(e.target.value))}><option value={1}>Unigram (N=1)</option><option value={2}>Bigram (N=2)</option><option value={3}>Trigram (N=3)</option></select></label><label>Smoothing<select value={alpha} onChange={e=>setAlpha(Number(e.target.value))}><option value={0}>None (MLE)</option><option value={.1}>Add 0.1</option><option value={1}>Add 1 (Laplace)</option></select></label></div></div></div><button className="primary" onClick={()=>run('ng')} disabled={!!busy}>{busy==='ng'?'Calculating…':'Calculate sequence probabilities'}</button>{ng&&<><div className="statistics"><div><strong>{ng.infinite?'∞':ng.perplexity?.toFixed(3)||'—'}</strong><span>perplexity</span></div><div><strong>{ng.sequence_probability!==undefined?(ng.sequence_probability===0?'0.0':ng.sequence_probability.toExponential(4)):'—'}</strong><span>sequence P(W)</span></div><div><strong>{ng.log_probability!==null?ng.log_probability:'—'}</strong><span>log P(W)</span></div><div><strong>{ng.vocabulary_size}</strong><span>vocabulary</span></div></div><p className="fineprint">{ng.notice} {ng.infinite?'An unseen event has zero probability without smoothing, so perplexity is infinite.':''}</p><div className="table-scroll capped"><table><thead><tr><th>Context</th><th>Next word</th><th>Count</th><th>Context count</th><th>Probability P(w|h)</th></tr></thead><tbody>{ng.events.map((r:any,i:number)=><tr key={i}><td>{r.context||'∅'}</td><td>{r.word}</td><td>{r.count}</td><td>{r.context_count}</td><td>{r.probability.toFixed(6)}</td></tr>)}</tbody></table></div></>}</Module><TfidfSection/><Module letter="K" title="LSA topic exploration" subtitle="Discover latent dimensions in a small collection of texts." open><p>Enter 3–30 documents, one per line, in the same language. LSA factorises the document–term TF-IDF matrix with singular value decomposition. The strongest terms help interpret each component.</p><div className="formula">X ≈ Uₖ Σₖ Vₖᵀ</div><label>Document collection<textarea rows={7} value={documents} onChange={e=>setDocuments(e.target.value)} maxLength={20000}/></label><button className="primary" onClick={()=>run('lsa')} disabled={!!busy}>{busy==='lsa'?'Computing…':'Explore latent topics'}</button>{ls&&<><p className="notice">{ls.notice}</p>{ls.topics&&<><div className="related-grid">{ls.topics.map((topic:any,i:number)=><div className="paper-panel" key={i}><h3>Component {topic.component}</h3><p className="fineprint">Variance explained: {pct(ls.explained_variance[i])}</p><div className="tokens">{topic.terms.map((t:any)=><span key={t.word} title={'Signed weight: '+t.weight}>{t.word} <small>{t.weight}</small></span>)}</div></div>)}</div><details><summary>Document vectors in the latent space</summary><div className="table-scroll"><table><thead><tr><th>Document</th>{ls.topics.map((t:any)=><th key={t.component}>C{t.component}</th>)}</tr></thead><tbody>{ls.coordinates.map((r:number[],i:number)=><tr key={i}><th>{i+1}</th>{r.map((n,j)=><td key={j}>{n.toFixed(4)}</td>)}</tr>)}</tbody></table></div></details></>}</>}</Module></>}

function Evaluation({data}:{data:any}){const [lang,setLang]=useState('en');const model=data.models.find((m:any)=>m.language===lang);const total=data.models.reduce((n:number,m:any)=>n+m.train_count,0);return <><div className="evaluation-intro"><div><div className="eyebrow">MEASURED, NOT INVENTED</div><h2>A small model.<br/>A real training pipeline.</h2><p>Language-specific TF-IDF features and linear support-vector machines. Hyperparameters selected on validation data, then evaluated on untouched test splits.</p></div><div className="training-total"><strong>{total.toLocaleString()}</strong><span>clean training headlines across {data.models.length} languages</span></div></div><div className="notice">Historical benchmark results are not a guarantee of accuracy on current news. Exact normalized duplicates and conflicting labels were removed before fitting. Near-duplicate events can remain.</div><label className="inline-label">Model language<select value={lang} onChange={e=>setLang(e.target.value)}><option value="en">English</option><option value="hi">हिन्दी</option><option value="mr">मराठी</option></select></label>{!model?<div className="empty"><h3>No trained model available.</h3><p>Follow the dataset and training instructions in README.md.</p></div>:<><div className="statistics"><div><strong>{pct(model.accuracy)}</strong><span>held-out accuracy</span></div><div><strong>{model.macro_f1.toFixed(4)}</strong><span>macro-F1</span></div><div><strong>{model.train_count.toLocaleString()}</strong><span>training examples</span></div><div><strong>{model.test_count.toLocaleString()}</strong><span>test examples</span></div></div><div className="analysis-grid"><div className="paper-panel"><h3>Training record</h3><dl><dt>Dataset</dt><dd>{model.dataset}</dd><dt>Algorithm</dt><dd>{model.algorithm}</dd><dt>Validation examples</dt><dd>{model.validation_count.toLocaleString()}</dd><dt>Validation-selected C</dt><dd>{model.selected_C}</dd><dt>Majority-class baseline</dt><dd>{pct(model.majority_baseline_accuracy)}</dd><dt>License</dt><dd>{model.license}</dd><dt>Trained</dt><dd>{new Date(model.trained_at).toLocaleString()}</dd></dl></div><div className="paper-panel"><h3>Supported categories</h3><div className="tokens">{model.labels.map((s:string)=><span key={s}>{s}</span>)}</div><p className="fineprint">Native dataset labels are preserved. No universal category mapping is used for prediction. Categories with fewer than 100 cleaned training examples are excluded.</p><h4>Validation selection</h4>{model.selection_trials.map((trial:any)=><p key={trial.C}>C={trial.C} · macro-F1 {trial.validation_macro_f1.toFixed(4)}</p>)}</div></div><Module letter="G" title="Confusion matrix" subtitle="Rows are actual labels. Columns are predicted labels." open><div className="table-scroll"><table className="confusion"><caption>{model.language} held-out confusion matrix · {model.test_count} examples</caption><thead><tr><th>Actual ↓ / predicted →</th>{model.labels.map((l:string)=><th key={l}>{l}</th>)}</tr></thead><tbody>{model.confusion_matrix.map((row:number[],i:number)=><tr key={i}><th>{model.labels[i]}</th>{row.map((n,j)=><td className={i===j?'diagonal':''} key={j}>{n}</td>)}</tr>)}</tbody></table></div></Module><details className="module"><summary><span className="module-letter">G</span><div><h2>Per-category performance & audit</h2><p>Inspect class support, precision, recall and data cleaning.</p></div></summary><div className="module-body"><div className="table-scroll"><table><thead><tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1</th><th>Test support</th></tr></thead><tbody>{model.labels.map((label:string)=>{const r=model.per_class[label];return <tr key={label}><th>{label}</th><td>{r.precision.toFixed(3)}</td><td>{r.recall.toFixed(3)}</td><td>{r['f1-score'].toFixed(3)}</td><td>{r.support}</td></tr>;})}</tbody></table></div><div className="table-scroll"><table><thead><tr><th>Split</th><th>Raw</th><th>Retained</th><th>Removed</th></tr></thead><tbody>{Object.entries(model.split_audit).map(([key,value])=>{const v=value as any;return <tr key={key}><th>{key}</th><td>{v.raw}</td><td>{v.retained}</td><td>{Object.entries(v.removed).map(([k,n])=>`${k}: ${n}`).join('; ')}{Object.keys(v.excluded_low_training_support||{}).length>0&&' · low-support labels: '+JSON.stringify(v.excluded_low_training_support)}</td></tr>;})}</tbody></table></div><p className="fineprint">Source revision: <code>{model.source_revision}</code><br/>Artifact SHA-256: <code className="break-word">{model.model_sha256}</code></p></div></details></>}</>}


function EmbeddingClassifierSection({language,mode}:{language:string;mode:string}){
  const [classifyText,setClassifyText]=useState(language==='hi'?'क्रिकेट टीम ने मैच जीतकर फाइनल में प्रवेश किया':language==='mr'?'क्रिकेट संघाने अंतिम सामन्यात विजय मिळवला':'Google and Microsoft researchers launch artificial intelligence software for mobile devices');
  const [result,setResult]=useState<any>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');
  async function runClassify(){
    setBusy(true);setError('');
    try{setResult(await api('classify-embeddings',{text:classifyText,language,mode}));}
    catch(e:any){setError(e.message);}
    finally{setBusy(false);}
  }
  return <div className="paper-panel" style={{marginTop:'28px'}}>
    <div className="section-heading" style={{margin:'0 0 12px'}}>
      <div>
        <h3>Text Classification using Word Embeddings (Assignment 4)</h3>
        <p>Averages 48-dim Word2Vec vectors across tokens and classifies via cosine similarity to category centroids.</p>
      </div>
    </div>
    <div className="filter-fields">
      <label style={{flex:3}}>Sentence to classify
        <input aria-label="Sentence to classify with embeddings" value={classifyText} onChange={e=>setClassifyText(e.target.value)} maxLength={300}/>
      </label>
      <button className="primary" onClick={runClassify} disabled={busy||!classifyText.trim()}>{busy?'Classifying…':'Classify with embeddings'}</button>
    </div>
    {error&&<p className="notice error" role="alert">{error}</p>}
    {result&&result.status==='available'&&<div className="analysis-grid" style={{marginTop:'16px'}}>
      <div>
        <p className="prediction">{result.predicted_category.toUpperCase()}<small>Cosine similarity: {result.similarity_score.toFixed(4)}</small></p>
        <p className="fineprint">{result.method}</p>
        <h4>Category similarities:</h4>
        {result.scores.map((s:any)=><div className="similarity-row" key={s.category}><span>{s.category}</span><div className="bar-track"><span style={{width:Math.max(0,s.similarity)*100+'%'}}/></div><code>{s.similarity.toFixed(4)}</code></div>)}
      </div>
      <div>
        <h4>Tokens & POS tags:</h4>
        <div className="table-scroll capped">
          <table><thead><tr><th>Token</th><th>POS Tag</th><th>In Word2Vec</th></tr></thead>
          <tbody>{result.pos_tokens.map((pt:any,i:number)=><tr key={i}><td>{pt.text}</td><td><code>{pt.tag}</code> ({pt.description})</td><td>{pt.in_vocab?'✓ Yes':'—'}</td></tr>)}</tbody>
          </table>
        </div>
        {result.vector_sample&&<p className="fineprint">Doc vector (first 6 dims): {result.vector_sample.slice(0,6).join(', ')}</p>}
      </div>
    </div>}
  </div>;
}

function OpinionLab(){
  const [docText,setDocText]=useState("The municipal library renovation is magnificent with spacious quiet study rooms and joyful readers.\nStaff at the community reading hall are extraordinarily helpful, friendly and supportive.\nDangerous open potholes and terrible road quality cause awful daily traffic jams and furious frustration.\nCommuters face immense anger due to reckless speeding trucks and broken traffic signals creating hazards.\nThe youth sports arena opened with an exciting cricket tournament and proud celebration.\nLocal sports facilities still need better maintenance and regular water supply.");
  const [singleText,setSingleText]=useState("The team celebrated their brilliant victory with great joy and happiness!");
  const [singleResult,setSingleResult]=useState<any>(null);
  const [singleBusy,setSingleBusy]=useState(false);
  const [miningResult,setMiningResult]=useState<any>(null);
  const [miningBusy,setMiningBusy]=useState(false);
  const [error,setError]=useState('');

  async function runSingle(){
    setSingleBusy(true);setError('');
    try{setSingleResult(await api('sentiment',{text:singleText,language:'en'}));}
    catch(e:any){setError(e.message);}
    finally{setSingleBusy(false);}
  }

  async function runMining(){
    setMiningBusy(true);setError('');
    try{
      const docs=docText.split('\n').map(x=>x.trim()).filter(Boolean);
      setMiningResult(await api('opinion-mining',{documents:docs}));
    }catch(e:any){setError(e.message);}
    finally{setMiningBusy(false);}
  }

  return <>
    {error&&<p className="notice error" role="alert">{error}</p>}
    <Module letter="10" title="Sentiment & emotion identification" subtitle="Detect positive/negative polarity and 6 emotional dimensions." open>
      <div className="analysis-grid">
        <label>Input text / headline
          <textarea rows={3} value={singleText} onChange={e=>setSingleText(e.target.value)} maxLength={2000}/>
        </label>
        <div style={{display:'flex',flexDirection:'column',justifyContent:'center'}}>
          <button className="primary" onClick={runSingle} disabled={singleBusy||!singleText.trim()}>{singleBusy?'Analyzing…':'Detect sentiment & emotions'}</button>
        </div>
      </div>
      {singleResult&&singleResult.status==='available'&&<div className="analysis-grid">
        <div className="paper-panel">
          <h3>Sentiment polarity</h3>
          <p className="prediction">{singleResult.label}<small>Compound score: {singleResult.compound>0?'+':''}{singleResult.compound.toFixed(4)}</small></p>
          <div className="similarity-row"><span>Positive</span><div className="bar-track"><span style={{width:Math.max(0,singleResult.polarity.pos)*100+'%'}}/></div><code>{(singleResult.polarity.pos*100).toFixed(1)}%</code></div>
          <div className="similarity-row"><span>Neutral</span><div className="bar-track"><span style={{width:Math.max(0,singleResult.polarity.neu)*100+'%'}}/></div><code>{(singleResult.polarity.neu*100).toFixed(1)}%</code></div>
          <div className="similarity-row"><span>Negative</span><div className="bar-track"><span style={{width:Math.max(0,singleResult.polarity.neg)*100+'%'}}/></div><code>{(singleResult.polarity.neg*100).toFixed(1)}%</code></div>
        </div>
        <div className="paper-panel">
          <h3>Emotion spectrum</h3>
          <p className="prediction">{singleResult.dominant_emotion}<small>Dominant emotion</small></p>
          {Object.entries(singleResult.emotions).map(([emo,val]:[string,any])=><div className="similarity-row" key={emo}><span style={{textTransform:'capitalize'}}>{emo}</span><div className="bar-track"><span style={{width:Math.max(0,Number(val))*100+'%'}}/></div><code>{(Number(val)*100).toFixed(1)}%</code></div>)}
        </div>
      </div>}
    </Module>

    <Module letter="10" title="Opinion mining with topic modelling" subtitle="Cluster citizen & reader opinions by topic and assess stance per topic." open>
      <p>Enter multiple opinion statements (one per line). Latent Semantic Indexing (LSI) discovers discussion themes, while VADER and emotion lexicons mine the stance per topic.</p>
      <label>Feedback & opinions collection (one per line)
        <textarea rows={7} value={docText} onChange={e=>setDocText(e.target.value)} maxLength={20000}/>
      </label>
      <button className="primary" onClick={runMining} disabled={miningBusy||!docText.trim()}>{miningBusy?'Mining opinions…':'Mine opinions by topic'}</button>
      {miningResult&&miningResult.status==='available'&&<>
        <div className="statistics">
          <div><strong>{miningResult.summary.total_opinions}</strong><span>Opinions</span></div>
          <div><strong>{miningResult.summary.overall_positive_pct}%</strong><span>Positive</span></div>
          <div><strong>{miningResult.summary.overall_negative_pct}%</strong><span>Negative</span></div>
          <div><strong>{miningResult.summary.topics_identified}</strong><span>Topics</span></div>
        </div>
        <div className="related-grid">
          {miningResult.topics.map((t:any)=><div className="paper-panel" key={t.topic_id}>
            <h3>Topic {t.topic_id}</h3>
            <p className="fineprint">Themes: {t.keywords.join(', ')}</p>
            <p className="prediction" style={{fontSize:'20px'}}>{t.sentiment_stance}<small>{t.positive_pct}% Pos · {t.negative_pct}% Neg · Emotion: {t.dominant_emotion}</small></p>
            <div className="similarity-row"><span>Stance</span><div className="bar-track"><span style={{width:Math.max(0,(t.avg_polarity+1)/2)*100+'%'}}/></div><code>{t.avg_polarity>0?'+':''}{t.avg_polarity.toFixed(2)}</code></div>
            <details style={{marginTop:'12px',fontSize:'12px'}}><summary>{t.opinions.length} sample opinions</summary>
              <ul style={{paddingLeft:'16px',margin:'8px 0'}}>{t.opinions.map((op:any,i:number)=><li key={i} style={{marginBottom:'6px'}}><strong>[{op.label}]</strong> {op.text}</li>)}</ul>
            </details>
          </div>)}
        </div>
      </>}
    </Module>
  </>;
}

function WebScraperLab({onSendToAnalysis}:{onSendToAnalysis:(text:string)=>void}){
  const [url,setUrl]=useState('https://en.wikipedia.org/wiki/Natural_language_processing');
  const [data,setData]=useState<any>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');

  async function handleScrape(targetUrl?:string){
    const toFetch = targetUrl || url;
    setBusy(true);setError('');
    try {
      const res = await api('scrape', { url: toFetch });
      setData(res);
    } catch(e:any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return <>
    <div className="section-heading">
      <div>
        <h2>Web Scraping & Online Data Gathering (Assignment 1)</h2>
        <p>Gather online articles and educational resources, extract clean body text, headings and hyperlinks, and feed them into the NLP pipeline.</p>
      </div>
      <Globe size={32}/>
    </div>
    <div className="lab-input">
      <div className="sample-pills">
        <span className="sample-label">Quick test sources:</span>
        <button type="button" className="sample-pill" onClick={()=>{setUrl('https://en.wikipedia.org/wiki/Natural_language_processing');handleScrape('https://en.wikipedia.org/wiki/Natural_language_processing');}}>Wikipedia: NLP</button>
        <button type="button" className="sample-pill" onClick={()=>{setUrl('https://example.com/news/sample');handleScrape('https://example.com/news/sample');}}>Tech News Story</button>
      </div>
      <div className="filter-fields">
        <label style={{flex:3}}>Web Page URL
          <input aria-label="Web page URL to scrape" value={url} onChange={e=>setUrl(e.target.value)} placeholder="https://..." />
        </label>
        <button className="primary" onClick={()=>handleScrape()} disabled={busy||!url.trim()}>
          <Globe size={16}/>{busy?'Scraping…':'Gather Web Data'}
        </button>
      </div>
    </div>
    {error&&<p className="notice error" role="alert">{error}</p>}
    {data&&data.status==='available'&&<>
      <div className="statistics">
        <div><strong>{data.stats.paragraph_count}</strong><span>paragraphs</span></div>
        <div><strong>{data.stats.word_count.toLocaleString()}</strong><span>words</span></div>
        <div><strong>{data.stats.heading_count}</strong><span>headings</span></div>
        <div><strong>{data.stats.link_count}</strong><span>links</span></div>
      </div>
      <p className="notice">{data.notice}</p>
      <div className="analysis-grid">
        <div className="paper-panel">
          <h3>Document Metadata</h3>
          <p className="prediction" style={{fontSize:'18px'}}>{data.title}<small>{data.domain}</small></p>
          {data.meta_description&&<p className="fineprint">{data.meta_description}</p>}
          <h4>Headings Discovered:</h4>
          <ul>{data.headings.map((h:string,i:number)=><li key={i}>{h}</li>)}</ul>
          <button className="primary" style={{marginTop:'16px'}} onClick={()=>onSendToAnalysis(data.text)}>
            <Play size={14}/> Send Extracted Text to NLP Lab
          </button>
        </div>
        <div className="paper-panel">
          <h3>Content Preview</h3>
          <div className="table-scroll capped" style={{maxHeight:'240px'}}>
            {data.paragraphs.map((p:string,i:number)=><p key={i} style={{marginBottom:'10px',fontSize:'13px'}}>{p}</p>)}
          </div>
          <h4>Hyperlinks Identified ({data.links.length}):</h4>
          <div className="tokens">
            {data.links.map((l:any,i:number)=><span key={i} title={l.url}>{l.text}</span>)}
          </div>
        </div>
      </div>
    </>}
  </>;
}

function TfidfSection(){
  const [docText,setDocText]=useState("Natural language processing uses statistical algorithms for computational linguistics.\nMachine learning algorithms improve natural language processing models.\nCricket players participate in regional championship tournaments.");
  const [res,setRes]=useState<any>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');

  async function runTfidf(){
    setBusy(true);setError('');
    try {
      const docs = docText.split('\n').map(x=>x.trim()).filter(Boolean);
      setRes(await api('tfidf', { documents: docs }));
    } catch(e:any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return <Module letter="6" title="TF-IDF Method (Term Frequency - Inverse Document Frequency)" subtitle="Calculate term weights across documents and evaluate cosine similarity." open>
    <div className="formula">
      TF(t, d) = count(t, d) / |d|<br/>
      <span>IDF(t) = ln((1 + N) / (1 + DF(t))) + 1 · TF-IDF(t, d) = TF(t, d) × IDF(t)</span>
    </div>
    <label>Documents (one per line, minimum 2 documents)
      <textarea rows={5} value={docText} onChange={e=>setDocText(e.target.value)} maxLength={20000}/>
    </label>
    <button className="primary" onClick={runTfidf} disabled={busy||!docText.trim()}>{busy?'Computing…':'Compute TF-IDF Matrix'}</button>
    {error&&<p className="notice error" role="alert">{error}</p>}
    {res&&res.status==='available'&&<>
      <p className="notice">{res.notice}</p>
      <div className="analysis-grid">
        <div className="paper-panel">
          <h3>Vocabulary & Inverse Document Frequency (IDF)</h3>
          <div className="table-scroll capped">
            <table><thead><tr><th>Term</th><th>Doc Freq (DF)</th><th>IDF Score</th></tr></thead>
            <tbody>{res.vocabulary.map((v:any)=><tr key={v.term}><td><strong>{v.term}</strong></td><td>{v.df} / {res.document_count}</td><td><code>{v.idf.toFixed(4)}</code></td></tr>)}</tbody>
            </table>
          </div>
        </div>
        <div className="paper-panel">
          <h3>Document Cosine Similarities in TF-IDF Space</h3>
          <div className="table-scroll">
            <table><thead><tr><th>Doc</th>{res.documents.map((d:any)=><th key={d.doc_id}>Doc {d.doc_id}</th>)}</tr></thead>
            <tbody>{res.similarity_matrix.map((row:number[],i:number)=><tr key={i}><th>Doc {i+1}</th>{row.map((val,j)=><td key={j} className={i===j?'diagonal':''}>{val.toFixed(4)}</td>)}</tr>)}</tbody>
            </table>
          </div>
          <h4>Top TF-IDF Terms per Document:</h4>
          {res.documents.map((d:any)=><p key={d.doc_id} style={{fontSize:'12px',margin:'6px 0'}}><strong>Doc {d.doc_id}:</strong> {d.top_terms.map((t:any)=>`${t.term} (${t.tfidf.toFixed(3)})`).join(', ')}</p>)}
        </div>
      </div>
    </>}
  </Module>;
}

function QaPipelineLab(){
  const [context,setContext]=useState("Google and Microsoft researchers announced breakthrough artificial intelligence software algorithms in California. The newly trained language models understand multilingual sentences with exceptional accuracy and joy. Dr. Rajesh Sharma led the natural language processing research group during this major achievement. However, some critics expressed serious concern and worry regarding computational power demands and energy costs.");
  const [question,setQuestion]=useState("Who led the natural language processing research group?");
  const [res,setRes]=useState<any>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');

  async function runQa(){
    setBusy(true);setError('');
    try {
      setRes(await api('qa', { context, question }));
    } catch(e:any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return <>
    <div className="section-heading">
      <div>
        <h2>Extractive Question Answering System (Assignment 9)</h2>
        <p>Part of the integrated NLP mini project pipeline: matches user queries to context documents using TF-IDF and Named Entity semantics.</p>
      </div>
      <HelpCircle size={32}/>
    </div>
    <div className="lab-input">
      <label>Context Passage
        <textarea rows={5} value={context} onChange={e=>setContext(e.target.value)} maxLength={20000}/>
      </label>
      <div className="filter-fields" style={{marginTop:'12px'}}>
        <label style={{flex:3}}>Question
          <input aria-label="Question to ask" value={question} onChange={e=>setQuestion(e.target.value)} maxLength={300}/>
        </label>
        <button className="primary" onClick={runQa} disabled={busy||!question.trim()||!context.trim()}>
          <Search size={16}/>{busy?'Finding Answer…':'Ask Question'}
        </button>
      </div>
    </div>
    {error&&<p className="notice error" role="alert">{error}</p>}
    {res&&res.status==='available'&&<div className="paper-panel" style={{marginTop:'20px'}}>
      <p className="fineprint">{res.method}</p>
      <h3>Best Extracted Answer:</h3>
      <p className="prediction" style={{fontSize:'18px',color:'var(--accent)'}}>
        “{res.best_answer}”
        <small>Relevance Confidence: {(res.confidence * 100).toFixed(1)}%</small>
      </p>
      <h4>Candidate Sentences Evaluated ({res.candidates.length}):</h4>
      <div className="table-scroll">
        <table><thead><tr><th>Rank</th><th>Candidate Sentence</th><th>Score</th><th>Matched Entities & Keywords</th></tr></thead>
        <tbody>{res.candidates.map((c:any,i:number)=><tr key={i}><td style={{fontWeight:'bold'}}>#{c.sentence_idx}</td><td>{c.sentence}</td><td><code>{c.score.toFixed(4)}</code></td><td>{c.matched_keywords.join(', ')} {c.entities.length>0?`[${c.entities.join('; ')}]`:''}</td></tr>)}</tbody>
        </table>
      </div>
    </div>}
  </>;
}
