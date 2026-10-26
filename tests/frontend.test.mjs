import test from 'node:test';
import assert from 'node:assert/strict';
import {toggleSaved,readList,safeExternalUrl,mergeArticles} from '../lib/storage.mjs';
test('bookmarks add and remove without mutating the input',()=>{const list=[];const a={id:'a',title:'Article'};const saved=toggleSaved(list,a);assert.equal(list.length,0);assert.deepEqual(saved,[a]);assert.deepEqual(toggleSaved(saved,a),[]);});
test('invalid storage is handled safely',()=>{assert.deepEqual(readList({getItem:()=>'{oops'},'x'),[]);assert.deepEqual(readList({getItem:()=>'{"bad":1}'},'x'),[]);assert.deepEqual(readList({getItem:()=>{throw Error('disabled')}},'x'),[]);});
test('unsafe external URL schemes and credentials are rejected',()=>{for(const s of ['javascript:alert(1)','data:text/html,hi','https://user:password@example.com','/relative',null])assert.equal(safeExternalUrl(s),null);assert.equal(safeExternalUrl('https://example.org/story'),'https://example.org/story');});
test('new metadata updates an existing article without duplicates',()=>{assert.deepEqual(mergeArticles([{id:'1',title:'old'}],[{id:'1',title:'new'},{id:'2',title:'other'}]),[{id:'1',title:'new'},{id:'2',title:'other'}]);});
