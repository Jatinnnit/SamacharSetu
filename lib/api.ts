export type Article = {id:string;title:string;description:string;body:string;language:string;publisher:string;timestamp:string|null;timestamp_kind:string;source_country:string;url:string|null;mode:string;topic?:string;prediction?:any};
export const languages:Record<string,string>={en:'English',hi:'हिन्दी',mr:'मराठी',bn:'বাংলা',ta:'தமிழ்',te:'తెలుగు',gu:'ગુજરાતી',kn:'ಕನ್ನಡ',ml:'മലയാളം',pa:'ਪੰਜਾਬੀ',or:'ଓଡ଼ିଆ',as:'অসমীয়া'};
export const topics=[['all','All topics'],['civic','Local & civic'],['politics','Politics'],['business','Economy'],['education','Education'],['health','Health'],['technology','Science & tech'],['environment','Environment'],['sports','Sports'],['entertainment','Culture']];
export async function api(path:string,data?:unknown,signal?:AbortSignal) {
  const response=await fetch('/api/'+path,{method:data?'POST':'GET',headers:data?{'Content-Type':'application/json'}:undefined,body:data?JSON.stringify(data):undefined,signal});
  let result; try {result=await response.json();}catch{throw new Error('The local NLP service is unavailable. Start the Python backend and try again.');}
  if(!response.ok) throw new Error(typeof result.detail==='string'?result.detail:'Please check the input and try again.');
  return result;
}
export const pct=(n:number)=>`${(n*100).toFixed(1)}%`;
export function dateLabel(article:Article){return article.timestamp?`${article.timestamp_kind==='first_seen'?'Seen ':''}${new Date(article.timestamp).toLocaleString(undefined,{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'})}`:'Undated learning example';}
