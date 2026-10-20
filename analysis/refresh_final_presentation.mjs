// Refresh code-generated figures in the final PowerPoint while preserving edits and slide order.
// Usage: RUNTIME_NODE_MODULES=<bundled packages> node analysis/refresh_final_presentation.mjs --output final-refreshed.pptx
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const packages=process.env.RUNTIME_NODE_MODULES;if(!packages)throw new Error('Set RUNTIME_NODE_MODULES to the bundled Node package directory');
const {PresentationFile,FileBlob}=await import(pathToFileURL(path.join(packages,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const args=process.argv.slice(2),at=args.indexOf('--output');if(at<0)throw new Error('Specify --output with a new destination');
const source=path.join(root,'output/presentation/lenin_stalin_forecast_final.pptx');
const output=path.resolve(args[at+1]);if(output===source)throw new Error('Use a new output file, then inspect it before replacing the final deck');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const images=new Map();
for(const dir of ['output/figures/en','output/figures/slides_en'])for(const file of await fs.readdir(path.join(root,dir)))if(file.endsWith('.png'))images.set(file.slice(0,-4),path.join(root,dir,file));
for(const slide of p.slides.items)for(const image of slide.images.items){
 const name=[...images.keys()].sort((a,b)=>b.length-a.length).find(n=>(image.alt||'').startsWith(n+' '));
 if(!name||['backtest_quality','monthly_coverage'].includes(name))continue; // preserve headerless cached test panels
 const frame=image.frame;image.replace({blob:new Uint8Array(await fs.readFile(images.get(name))),contentType:'image/png',fit:'contain',alt:image.alt});image.frame=frame;
}
await(await PresentationFile.exportPptx(p)).save(output);console.log(output);
