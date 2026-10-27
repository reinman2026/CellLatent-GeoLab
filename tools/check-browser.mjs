import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {tmpdir} from 'node:os';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createServer} from 'node:http';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),out=path.join(root,'.checks');
const profile=path.join(tmpdir(),'cdx-ui-'+Date.now());await fs.mkdir(profile,{recursive:true});
const server=createServer(async(req,res)=>{try{const requested=decodeURIComponent(new URL(req.url,'http://localhost').pathname),file=path.resolve(root,'.'+(requested==='/'?'/dist/index.html':requested));if(!file.startsWith(root+path.sep))throw Error('Invalid path');res.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json'})[path.extname(file)]??'application/octet-stream');res.end(await fs.readFile(file));}catch{res.statusCode=404;res.end('Not found');}});
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
const browser=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',['--headless=new','--disable-gpu','--in-process-gpu','--disable-gpu-sandbox','--no-sandbox','--disable-dev-shm-usage','--disable-software-rasterizer','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{windowsHide:true,stdio:['ignore','pipe','pipe']});
let launchStderr='';browser.stderr.on('data',chunk=>{launchStderr+=chunk.toString();});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const fetchJson=async url=>{let last;for(let i=0;i<30;i++){try{const response=await fetch(url,{signal:AbortSignal.timeout(1500)});if(!response.ok)throw Error(`HTTP ${response.status}`);return await response.json();}catch(error){last=error;await pause(200);}}throw Error(`DevTools endpoint unavailable: ${last?.message??last}`);};
let socket;
try{
  let port;for(let i=0;i<80;i++){try{port=(await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break;}catch{}await pause(150);}if(!port)throw Error('Browser debugging endpoint unavailable');
  const version=await fetchJson(`http://127.0.0.1:${port}/json/version`);socket=new WebSocket(version.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('Browser socket timeout')),8000);socket.onopen=()=>{clearTimeout(timer);resolve();};socket.onerror=e=>{clearTimeout(timer);reject(e);};});
  let serial=0;const pending=new Map(),errors=[];
  socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.exception?.description??m.params.exceptionDetails.text);else if(m.method==='Runtime.consoleAPICalled'&&m.params.type==='error')errors.push(JSON.stringify(m.params.args));};
  const call=(method,params={},sessionId=null)=>new Promise((resolve,reject)=>{const id=++serial,timer=setTimeout(()=>{pending.delete(id);reject(Error('Browser command timeout: '+method));},12000);pending.set(id,{resolve:v=>{clearTimeout(timer);resolve(v);},reject:e=>{clearTimeout(timer);reject(e);}});socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});
  const {targetId}=await call('Target.createTarget',{url:'about:blank'}),{sessionId}=await call('Target.attachToTarget',{targetId,flatten:true});
  const cmd=(m,p={})=>call(m,p,sessionId);await cmd('Page.enable');await cmd('Runtime.enable');
  const evaluate=async expression=>{const r=await cmd('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.exception?.description??r.exceptionDetails.text);return r.result.value;};
  const waitFor=async expression=>{for(let i=0;i<1000;i++){if(await evaluate(expression))return;await pause(50);}throw Error('Timed out: '+expression+'\n'+errors.join('\n'));};
  const base=`http://127.0.0.1:${server.address().port}/dist/index.html`;
  const navigate=async(url,route)=>{await cmd('Page.navigate',{url});await waitFor(`window.__latentLab?.route === ${JSON.stringify(route)}`);await pause(100);};
  const click=id=>evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
  const set=(id,value,event='change')=>evaluate(`document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(String(value))};document.getElementById(${JSON.stringify(id)}).dispatchEvent(new Event(${JSON.stringify(event)}))`);
  const state='window.__latentLab.state()';
  const shot=async(name)=>{const r=await cmd('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path.join(out,name+'.png'),Buffer.from(r.data,'base64'));};
  await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false});
  await navigate(base,'replicas');
  assert.equal(await evaluate(`${state}.bundle.models.length`),5);
  assert.equal(await evaluate("document.querySelectorAll('.rp-three').length"),5);
  assert.equal(await evaluate("document.querySelectorAll('.rp-three canvas').length"),15);
  assert.equal(await evaluate(`${state}.explanationModel.index`),0);
  assert(await evaluate("document.querySelector('.rp-row-label strong').textContent.includes('第一行代表模型 1')"));
  assert(await evaluate("document.querySelector('.rp-geometry-note').textContent.includes('Pearson') && document.querySelector('.rp-geometry-note').textContent.includes('同一批观测')"));
  assert(await evaluate("document.querySelector('.rp-reading-map').textContent.includes('同一批样本') && document.querySelector('.rp-reading-map').textContent.includes('几何相似') && document.querySelector('.rp-reading-map').textContent.includes('训练收敛')"));
  assert(await evaluate("document.querySelector('.rp-parameter-note').textContent.includes('第一行模型 1 的具体参数讲解') && document.querySelector('.rp-parameter-note').textContent.includes('n_hidden')"));
  await shot('replicas-desktop-five-by-three');
  await cmd('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert(await evaluate('document.documentElement.scrollWidth <= innerWidth + 1'));
  await shot('replicas-mobile-five-by-three');
  await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false});
  await navigate(base+'#overview','overview');
  assert.equal(await evaluate('window.__latentLab.model.input'),6);
  assert.equal(await evaluate('window.__latentLab.model.dim'),3);
  assert(await evaluate("['sp-train-one','sp-train','sp-auto-fit','sp-stop','sp-reset'].every(id=>document.getElementById(id).closest('.fit-toolbar'))"));
  await set('sp-cell',8);assert.equal(await evaluate(`${state}.selected`),8);
  await click('sp-next');assert.equal(await evaluate(`${state}.selected`),9);
  const prediction=await evaluate(`JSON.stringify(${state}.rows[9].prediction)`);
  await click('sp-selected-ellipsoid');await set('sp-contour',2.795483482915);await click('sp-all-ellipsoids');
  assert.equal(await evaluate(`JSON.stringify(${state}.rows[9].prediction)`),prediction);
  const yaw=await evaluate(`${state}.yaw`);
  await evaluate("document.getElementById('sp-latent-chart').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}))");
  assert.notEqual(await evaluate(`${state}.yaw`),yaw);
  await evaluate("location.hash='encoder'");await waitFor("window.__latentLab.route==='encoder'");
  assert.equal(await evaluate(`${state}.selected`),9);
  await evaluate("document.querySelector('[data-sp-unit=\"3\"]').click()");
  assert.equal(await evaluate(`${state}.unit`),3);
  assert.equal(await evaluate("document.querySelectorAll('#sp-products tbody tr').length"),6);
  await set('sp-repr','sample');await click('sp-resample');
  await evaluate("location.hash='training'");await waitFor("window.__latentLab.route==='training'");
  const initial=await evaluate(`${state}.model.evaluate({lambda:.05}).loss`);
  await set('sp-epochs',100);await click('sp-train');await waitFor(`${state}.model.epoch===100 && !${state}.busy`);
  assert((await evaluate(`${state}.model.evaluate({lambda:.05}).loss`))<initial);
  await evaluate("location.hash='space'");await waitFor("window.__latentLab.route==='space'");
  assert.equal(await evaluate(`${state}.model.epoch`),100);assert.equal(await evaluate(`${state}.umap.length`),24);assert.equal(await evaluate("document.querySelectorAll('#sp-kinetics-table tbody tr').length"),3);await click('sp-cloud');
  assert.equal(await evaluate(`${state}.showCloud`),true);
  await evaluate("location.hash='kinetics'");await waitFor("window.__latentLab.route==='kinetics'");
  await set('kl-cell',8);await set('kl-gene',2);assert.equal(await evaluate(`${state}.gene`),2);
  assert(await evaluate("['kl-mode','kl-step','kl-train','kl-auto-fit','kl-pause','kl-reset'].every(id=>document.getElementById(id).closest('.fit-toolbar'))"));
  assert.equal(await evaluate("document.querySelectorAll('#kl-comparison tbody tr').length"),6);
  const timesK=await evaluate(`JSON.stringify(${state}.rows[8].time)`),paramsK=await evaluate(`JSON.stringify(${state}.model.p)`);
  await set('kl-cursor',0,'input');assert.equal(await evaluate(`${state}.cursor`),0);assert.equal(await evaluate(`JSON.stringify(${state}.model.p)`),paramsK);
  await set('kl-beta',1.2,'input');assert.equal(await evaluate(`JSON.stringify(${state}.rows[8].time)`),timesK);
  await set('kl-time-bias',.6,'input');assert.notEqual(await evaluate(`JSON.stringify(${state}.rows[8].time)`),timesK);
  await click('kl-reset');await set('kl-mode','rates');const fixedTime=await evaluate(`JSON.stringify(${state}.rows[8].time)`);await click('kl-step');assert.equal(await evaluate(`JSON.stringify(${state}.rows[8].time)`),fixedTime);
  await set('kl-mode','time');const fixedRates=await evaluate(`JSON.stringify(${state}.model.rates())`);await click('kl-step');assert.equal(await evaluate(`JSON.stringify(${state}.model.rates())`),fixedRates);
  await set('kl-mode','joint');await set('kl-epochs',100);const initialK=await evaluate(`${state}.loss.loss`);await click('kl-train');await waitFor(`${state}.model.epoch===102 && !${state}.busy`);assert((await evaluate(`${state}.loss.loss`))<initialK);assert(await evaluate(`${state}.tankSnapshots[0].epoch===0 && ${state}.tankSnapshots.some(s=>s.epoch===10) && ${state}.tankSnapshots.at(-1).epoch===102`));
  await click('kl-auto-fit');await waitFor(`${state}.autoSession?.attempts>=2`);await click('kl-pause');assert.equal(await evaluate(`${state}.autoResult.reason`),'paused');assert(await evaluate(`Math.abs(${state}.loss.loss-${state}.autoResult.bestLoss)<1e-10`));
  assert(await evaluate("!document.getElementById('kl-best-step').hidden"));
  const routes=['overview','encoder','training','space','kinetics','construction'];
  const replicaRoutes=['replicas','stability','geometry','kinetics','replica-compare'];
  const paperRoutes=['paper'];
  for(const width of [1440,390,320]){
    await cmd('Emulation.setDeviceMetricsOverride',{width,height:width===1440?1050:844,deviceScaleFactor:1,mobile:width!==1440});
    for(const route of routes){
      await navigate(base+'#'+route,route);
      assert(await evaluate('document.documentElement.scrollWidth <= innerWidth + 1'),`Overflow ${route}/${width}`);
      assert.equal(await evaluate(`${state}.model.input`),6);assert.equal(await evaluate(`${state}.model.dim`),3);
      if(route==='construction')assert(await evaluate("document.getElementById('sp-sim-framework') && document.querySelectorAll('#sp-sim-framework .sp-framework-node').length===5"));
      if(route!=='kinetics')assert(await evaluate("document.getElementById('sp-input-chart').getBoundingClientRect().top < document.getElementById('sp-latent-chart').getBoundingClientRect().top && document.getElementById('sp-latent-chart').getBoundingClientRect().top < document.getElementById('sp-output-chart').getBoundingClientRect().top"));
      await shot('spatial-'+route+'-'+width);
      if(route==='kinetics'){
        await evaluate("document.getElementById('kl-trajectory').scrollIntoView({block:'center'})");await shot('kinetics-trajectory-'+width);
        await evaluate("document.getElementById('kl-training-panel').scrollIntoView({block:'start'})");await shot('kinetics-training-'+width);
      }
    }
    for(const route of replicaRoutes){
      await navigate(base+'#'+route,route);
      assert(await evaluate('document.documentElement.scrollWidth <= innerWidth + 1'),`Overflow ${route}/${width}`);
      if(route==='replicas')assert.equal(await evaluate("document.querySelectorAll('.rp-three canvas').length"),15);
      if(route==='stability')assert.equal(await evaluate("document.querySelectorAll('#st-grid canvas').length"),5);
      if(route==='replica-compare')assert.equal(await evaluate("document.querySelectorAll('.rp-three canvas').length"),9);
      if(route==='kinetics')assert(await evaluate("document.querySelectorAll('#kl-replica-preview canvas').length===3 && document.getElementById('kl-rp-latent').tabIndex>=0 && document.getElementById('kl-tank-history') && document.querySelector('#kl-training-panel').textContent.includes('教学模型实时训练过程')"));
      await shot('replica-'+route+'-'+width);
    }
    for(const route of paperRoutes){
      await navigate(base+'#'+route,route);
      assert(await evaluate('document.documentElement.scrollWidth <= innerWidth + 1'),`Overflow ${route}/${width}`);
      assert(await evaluate("document.querySelectorAll('.paper-net-card').length===3 && document.querySelectorAll('.paper-edges tbody tr').length===6"));
      await click('paper-reset');const before=await evaluate(`${state}.current[0]`);const summaryBefore=await evaluate("document.getElementById('paper-summary').textContent");await set('paper-step',50,'input');assert.notEqual(await evaluate(`${state}.current[0]`),before);assert.notEqual(await evaluate("document.getElementById('paper-summary').textContent"),summaryBefore);
      await shot('paper-'+width);
    }
  }
  const offline=pathToFileURL(path.join(root,'离线版.html')).href;
  for(const route of routes)await navigate(offline+'#'+route,route);
  for(const route of replicaRoutes)await navigate(offline+'#'+route,route);
  for(const route of paperRoutes){await navigate(offline+'#'+route,route);assert(await evaluate("document.querySelectorAll('.paper-net-card').length===3 && document.querySelectorAll('.paper-edges tbody tr').length===6"));}
  assert.equal(errors.length,0,errors.join('\n'));
  const result={passed:true,browser:version.Browser,routes:[...new Set([...routes,...replicaRoutes,...paperRoutes])],widths:[1440,390,320],shape:[6,3,6],consoleErrors:errors,checks:['three-step reading map: fixed samples, geometry similarity and training convergence','five-row by three-column representative page with first-row parameter explanation','original three-space teaching page with linked 6D observation, rotatable 3D latent view and 6D reconstruction','simulation final framework diagram','paper experiment flow and prior-to-learned network comparison','paper edge interpolation slider and edge table','five geometry representatives on the geometry pages','shared principal-axis screen rotation for axes, hull, MVEE and points','model-1 z1/z2/z3 fixed cell ordering','sample-order selector','shared flow toggle and state across all representative latent spaces','independent coordinate-axis, confidence-ellipsoid, convex-hull, MVEE and point layers','linked cell selection','ellipsoid scope and contour preserve decoding','keyboard rotation','two-pool kinetics table','illustrative 2D velocity display','six input products','posterior sampling','100 real direct-reconstruction updates','shared original training state','kinetics observation cursor','manual beta and time bias independence','rates-only and time-only frozen groups','100 joint ODE training updates','teaching-model reservoir snapshots at 0/10/final updates','offline routes','desktop and mobile layouts']};
  await fs.writeFile(path.join(out,'browser-check.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));await call('Browser.close');
}catch(error){if(launchStderr.trim())error.message+=`\nChrome stderr:\n${launchStderr.trim()}`;throw error;}finally{socket?.close();if(browser.exitCode===null)browser.kill();server.close();}
