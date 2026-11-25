/** Verify the pinned demo artifact before exposing any inference assets. */
import {readFile,writeFile,mkdir,cp} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(path.join(root,'apps/web/package.json'));
const AdmZip=require('adm-zip');
const spec=JSON.parse(await readFile(path.join(root,'configs/i1_demo_artifact.json'),'utf8'));
const directory=path.join(root,'artifacts/i1');
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const safe=name=>{if(!/^[a-zA-Z0-9_./-]+$/.test(name)||name.startsWith('/')||name.split('/').includes('..'))throw Error('Unsafe artifact member');return path.join(directory,name);};
let archive;
try{archive=await readFile(path.join(directory,'GridLens_I1_Demo.zip'));}catch{
  const response=await fetch(spec.url,{signal:AbortSignal.timeout(60000)});
  if(!response.ok)throw Error('Unable to download pinned I1 artifact');
  if(Number(response.headers.get('content-length'))>spec.size_bytes)throw Error('Oversized artifact');
  archive=Buffer.from(await response.arrayBuffer());
}
if(archive.length!==spec.size_bytes||digest(archive)!==spec.sha256)throw Error('I1 archive checksum mismatch');
await mkdir(directory,{recursive:true});
const zip=new AdmZip(archive);let size=0;
for(const entry of zip.getEntries()){
  const target=safe(entry.entryName);
  if(entry.isDirectory)continue;
  size+=entry.header.size;if(size>10000000)throw Error('Oversized expanded artifact');
  await mkdir(path.dirname(target),{recursive:true});await writeFile(target,entry.getData());
}
const manifestBytes=await readFile(path.join(directory,'manifest.json'));
if(digest(manifestBytes)!==spec.manifest_sha256)throw Error('I1 manifest checksum mismatch');
const manifest=JSON.parse(manifestBytes);
if(manifest.kind!=='gridlens_i1_demo_v1')throw Error('Unexpected artifact kind');
const names=new Set();
for(const entry of manifest.files){
  if(names.has(entry.path))throw Error('Duplicate artifact member');names.add(entry.path);
  const bytes=await readFile(safe(entry.path));
  if(bytes.length!==entry.size_bytes||digest(bytes)!==entry.sha256)throw Error(`Corrupt artifact: ${entry.path}`);
}
const catalog=JSON.parse(await readFile(path.join(directory,'client/catalog.json'),'utf8'));
if(catalog.catalog_id!==spec.catalog_id)throw Error('Unexpected catalog identity');
await cp(path.join(directory,'client'),path.join(root,'apps/web/public/inference'),{recursive:true});
const ort=path.dirname(require.resolve('onnxruntime-web/wasm'));
await mkdir(path.join(root,'apps/web/public/ort'),{recursive:true});
for(const file of ['ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm'])await cp(path.join(ort,file),path.join(root,'apps/web/public/ort',file));
console.log(`Verified I1 assets: ${catalog.cases.length} validation origins; model results are computed on demand.`);
