import {spawn} from 'node:child_process';
import {mkdir,readFile,writeFile,stat} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {tmpdir} from 'node:os';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),out=path.join(root,'.checks');
await mkdir(out,{recursive:true});
const target=pathToFileURL(path.join(root,'离线版.html')).href;
const report=[];
for(const [name,width,height,route] of [['spatial-desktop',1440,1250,'overview'],['spatial-mobile',390,1800,'overview'],['spatial-training',1440,1600,'training']]){
  const file=path.join(out,name+'.png'),profile=path.join(tmpdir(),'cdx-sp-'+Date.now());
  const args=['--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--hide-scrollbars','--virtual-time-budget=1500','--timeout=12000',`--user-data-dir=${profile}`,`--window-size=${width},${height}`,`--screenshot=${file}`,'--dump-dom',target+'#'+route];
  const result=await new Promise((resolve,reject)=>{
    const child=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',args,{windowsHide:true});let stdout='',stderr='';
    const timer=setTimeout(()=>{child.kill();reject(Error('Headless render timed out'));},25000);
    child.stdout.on('data',d=>stdout+=d);child.stderr.on('data',d=>stderr+=d);child.on('error',reject);child.on('close',code=>{clearTimeout(timer);resolve({code,stdout,stderr});});
  });
  let size=0;try{size=(await stat(file)).size;}catch{}
  await writeFile(path.join(out,name+'.dom.html'),result.stdout);
  const item={name,width,height,exitCode:result.code,bytes:size,rendered:result.stdout.includes('id="sp-input-chart"')&&result.stdout.includes('id="sp-output-vector"'),stderr:result.stderr.slice(-600)};report.push(item);console.log(JSON.stringify(item));
  if(!size||!item.rendered)throw Error('Headless renderer did not complete '+name);
}
await writeFile(path.join(out,'spatial-render-check.json'),JSON.stringify(report,null,2));
