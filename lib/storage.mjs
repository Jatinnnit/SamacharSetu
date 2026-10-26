export function readList(storage, key) {
  try { const result=JSON.parse(storage.getItem(key)||'[]'); return Array.isArray(result)?result:[]; } catch { return []; }
}
export function toggleSaved(list, article) {
  return list.some(item=>item.id===article.id)?list.filter(item=>item.id!==article.id):[article,...list].slice(0,300);
}
export function safeExternalUrl(value) {
  try {const url=new URL(value);return ['https:','http:'].includes(url.protocol)&&!url.username&&!url.password?url.href:null;}catch{return null;}
}
export function mergeArticles(previous, incoming) {
  return [...new Map([...previous,...incoming].map(item=>[item.id,item])).values()].slice(-400);
}
