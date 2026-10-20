// Rasterize saved SVG outputs after their text labels are updated.
// Run with the bundled Node runtime; set RUNTIME_NODE_MODULES to its package directory.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const packages=process.env.RUNTIME_NODE_MODULES;
if(!packages)throw new Error('Set RUNTIME_NODE_MODULES to the bundled Node package directory');
const require=createRequire(pathToFileURL(path.join(packages,'@oai/artifact-tool/package.json')));
const {Canvas,loadImage}=require('skia-canvas');
const dir=path.resolve(process.argv[2]||'output/figures/en');
let count=0;
for(const name of (await fs.readdir(dir)).filter(x=>x.endsWith('.svg'))){
 const svg=await fs.readFile(path.join(dir,name),'utf8');
 if(/[\u2012-\u2015]/u.test(svg))throw new Error('Long dash remains in '+name);
 const view=svg.match(/viewBox="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"/);
 if(!view)throw new Error('Missing viewBox: '+name);
 const w=Math.ceil(Number(view[3])*2.5),h=Math.ceil(Number(view[4])*2.5);
 const image=await loadImage(Buffer.from(svg));const canvas=new Canvas(w,h);
 canvas.getContext('2d').drawImage(image,0,0,w,h);
 await canvas.toFile(path.join(dir,name.replace(/\.svg$/,'.png')));count++;
}
console.log('Rendered '+count+' SVGs without long dashes');
