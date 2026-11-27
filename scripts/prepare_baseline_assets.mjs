/** Retrieve the approved I0 archive; never overwrite a different existing file. */
import {readFile,writeFile,mkdir,access} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(path.join(root,'apps/web/package.json'));
const AdmZip=require('adm-zip');
const spec=JSON.parse(await readFile(path.join(root,'configs/i0_public_artifact_v1.json'),'utf8'));
const output=path.resolve(process.argv[2]??path.join(root,'reports/local/i0-public'));
const response=await fetch(spec.url,{signal:AbortSignal.timeout(60000)});
if(!response.ok)throw Error('Pinned baseline archive unavailable');
let length=0;const parts=[];
for await(const part of response.body){length+=part.length;if(length>spec.size_bytes)throw Error('Oversized baseline archive');parts.push(part);}
const bytes=Buffer.concat(parts);
const digest=value=>createHash('sha256').update(value).digest('hex');
if(length!==spec.size_bytes||digest(bytes)!==spec.sha256)throw Error('Baseline archive checksum mismatch');
const entries=new AdmZip(bytes).getEntries();const files=[];let expanded=0;
for(const entry of entries){
 const name=entry.entryName;
 if(!/^[a-zA-Z0-9_./-]+$/.test(name)||name.startsWith('/')||name.split('/').includes('..')||((entry.header.attr>>>16)&0xf000)===0xa000)throw Error('Unsafe baseline member');
 if(entry.isDirectory)continue;
 expanded+=entry.header.size;if(expanded>10000000)throw Error('Oversized expanded baseline');
 const destination=path.join(output,name);const value=entry.getData();
 try{await access(destination);if(digest(await readFile(destination))!==digest(value))throw Error(`Existing baseline file differs: ${name}`);}catch(error){if(error.code!=='ENOENT')throw error;}
 files.push({destination,value});
}
for(const {destination,value} of files){await mkdir(path.dirname(destination),{recursive:true});await writeFile(destination,value,{flag:'wx'}).catch(error=>{if(error.code!=='EEXIST')throw error;});}
console.log(`Verified public baseline archive and ${files.length} files at ${output}`);
