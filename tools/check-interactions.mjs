// DOM/canvas test doubles exercise the offline script without launching a
// browser. This catches route/handler failures, not visual layout defects.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const ctx=new Proxy({measureText:s=>({width:String(s).length*7}),createRadialGradient:()=>({addColorStop:()=>{}})},{get:(t,k)=>k in t?t[k]:(()=>{}),set:(t,k,v)=>(t[k]=v,true)});
class Element {
  constructor(tag,attrs={}){this.tagName=tag;this.attrs=attrs;this.children=[];this.dataset={};this.disabled=false;this.checked='checked' in attrs;this._value=attrs.value;this._text='';this.clientWidth=620;this.clientHeight=360;this.classList={toggle:()=>{},add:()=>{},remove:()=>{}};for(const [k,v] of Object.entries(attrs))if(k.startsWith('data-'))this.dataset[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=v;}
  all(){return this.children.flatMap(c=>[c,...c.all()]);}
  set innerHTML(html){this.children=[];this._text='';parse(String(html),this);}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
  set textContent(t){this._text=String(t);this.children=[];}
  get value(){if(this._value!==undefined)return this._value;const options=this.all().filter(c=>c.tagName==='option');const o=options.find(c=>'selected' in c.attrs)??options[0];return o?(o.attrs.value??o.textContent):'';}
  set value(v){this._value=String(v);}
  getAttribute(k){return this.attrs[k];}setAttribute(k,v){this.attrs[k]=v;}
  querySelectorAll(selector){const a=selector.match(/^\[([\w-]+)(?:=["']?(.*?)["']?)?\]$/);return this.all().filter(c=>a?(a[1] in c.attrs&&(a[2]===undefined||c.attrs[a[1]]===a[2])):c.tagName===selector);}
  getContext(){return ctx;}getBoundingClientRect(){return {left:0,top:0,width:this.clientWidth,height:this.clientHeight};}
  click(){if(!this.disabled){if(this.attrs.type==='checkbox'){this.checked=!this.checked;this.onchange?.();}this.onclick?.({currentTarget:this,clientX:0,clientY:0});}}
}
function parse(html,root){const stack=[root];const tokens=html.match(/<[^>]*>|[^<]+/g)??[];for(const token of tokens){if(token.startsWith('</')){const name=token.slice(2,-1).trim();let n=stack.length-1;while(n>0&&stack[n].tagName!==name)n--;if(n>0)stack.length=n;}else if(/^<[a-z]/i.test(token)){const m=token.match(/^<([\w-]+)([\s\S]*?)\/?\s*>$/);if(!m)continue;const attrs={};for(const a of m[2].matchAll(/([\w:-]+)(?:="([^"]*)"|='([^']*)'|=([^\s>]+))?/g))attrs[a[1]]=a[2]??a[3]??a[4]??'';const el=new Element(m[1],attrs);stack.at(-1).children.push(el);if(!['input','br','hr','meta','link','img'].includes(m[1]))stack.push(el);}else if(!token.startsWith('<'))stack.at(-1)._text+=token;}}
const html=readFileSync(new URL('../离线版.html',import.meta.url),'utf8'),script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const document=new Element('document');document.innerHTML=html.split('<script>')[0].replace(/<style>[\s\S]*?<\/style>/g,'');document.getElementById=id=>document.all().find(e=>e.attrs.id===id)??null;
const listeners={},location={_hash:'#overview',get hash(){return this._hash;},set hash(v){this._hash=v.startsWith('#')?v:'#'+v;listeners.hashchange?.();}};
const window={document,location,devicePixelRatio:1,scrollTo:()=>{},addEventListener:(name,callback)=>listeners[name]=callback};
const context=vm.createContext({window,document,location,devicePixelRatio:1,setTimeout,clearTimeout,console});vm.runInContext(script,context,{timeout:10000});
const $=id=>{const el=document.getElementById(id);assert(el,'Missing rendered element '+id);return el;};
const change=(id,value)=>{$(id).value=value;$(id).onchange?.();};const state=()=>window.__latentLab.state();
const wait=async condition=>{for(let i=0;i<2000;i++){if(condition())return;await new Promise(r=>setTimeout(r,25));}throw Error('Timed out');};
assert.equal(window.__latentLab.route,'overview');assert.equal(state().model.input,12);assert.equal(state().model.dim,3);assert.equal(state().rows[0].prediction.length,12);
change('sp-cell',8);assert.equal(state().selected,8);$('sp-next').click();assert.equal(state().selected,9);$('sp-prev').click();
const data=JSON.stringify(state().model.data);$('sp-delta').value=.4;$('sp-delta').oninput();assert.equal(state().delta,.4);assert.equal(JSON.stringify(state().model.data),data);$('sp-restore').click();
const prediction=JSON.stringify(state().rows[8].prediction),mean=JSON.stringify(state().rows[8].mu),variance=JSON.stringify(state().rows[8].variance);
$('sp-selected-ellipsoid').click();assert.equal(state().ellipsoidScope,'selected');
change('sp-contour',2.795483482915);assert.equal(state().contourScale,2.795483482915);
const yaw=state().yaw;$('sp-latent-chart').onkeydown({key:'ArrowLeft',preventDefault(){}});assert.notEqual(state().yaw,yaw);
$('sp-all-ellipsoids').click();assert.equal(state().ellipsoidScope,'all');
assert.equal(JSON.stringify(state().rows[8].prediction),prediction);assert.equal(JSON.stringify(state().rows[8].mu),mean);assert.equal(JSON.stringify(state().rows[8].variance),variance);
change('sp-repr','sample');const oldSample=JSON.stringify(state().rows[8].z);$('sp-resample').click();assert.notEqual(JSON.stringify(state().rows[8].z),oldSample);assert.equal(JSON.stringify(state().rows[8].mu),mean);assert.equal(JSON.stringify(state().rows[8].variance),variance);
location.hash='encoder';assert.equal(state().selected,8);for(const b of document.querySelectorAll('[data-sp-step]'))b.click();document.querySelectorAll('[data-sp-unit]')[3].click();assert.equal(state().unit,3);
change('sp-repr','mean');location.hash='training';change('sp-epochs',100);const before=state().model.evaluate({lambda:.05}).loss;$('sp-train').click();await wait(()=>state().model.epoch===100&&!state().busy);assert(state().model.evaluate({lambda:.05}).loss<before);assert.equal(JSON.stringify(state().model.data),data);
$('sp-train').click();await wait(()=>state().model.epoch>100);location.hash='space';const stopped=state().model.epoch;await new Promise(r=>setTimeout(r,100));assert.equal(state().model.epoch,stopped);$('sp-cloud').click();assert(state().showCloud);
assert.equal(state().umap.length,300);assert(state().umap.every(p=>p.xy.length===2&&p.velocity.length===2&&p.velocity.every(Number.isFinite)));assert.equal(state().model.kinetics.length,6);assert.equal($('sp-kinetics-table').querySelectorAll('tr').length,7);
for(const route of ['overview','training','space','encoder','construction']){location.hash=route;assert.equal(window.__latentLab.route,route);assert.equal(state().model.epoch,stopped);assert.equal(state().model.input,12);assert.equal(state().model.dim,3);listeners.resize();await new Promise(r=>setTimeout(r,120));}
location.hash='posterior';assert.equal(window.__latentLab.route,'space');location.hash='network';assert.equal(window.__latentLab.route,'encoder');
location.hash='training';$('sp-reset').click();assert.equal(state().model.epoch,0);
for(const route of ['overview','encoder','training','space','construction']){
  location.hash=route;const toolbar=document.all().find(e=>e.attrs.class?.split(' ').includes('fit-toolbar'));assert(toolbar);
  for(const id of ['sp-train-one','sp-train','sp-auto-fit','sp-stop','sp-reset']){assert(toolbar.all().includes($(id)),id+' must be in top toolbar');assert.equal(document.all().filter(e=>e.attrs.id===id).length,1);}
}
location.hash='overview';change('sp-epochs',100);const autoStart=state().model.evaluate({lambda:.05}).loss;$('sp-auto-fit').click();await wait(()=>!state().busy&&state().autoResult);
assert(state().autoResult.attempts<=400);assert(state().autoResult.attempts>=40);assert(state().autoResult.bestLoss<autoStart);assert(Math.abs(state().model.assess({lambda:.05}).loss-state().autoResult.bestLoss)<1e-10);
assert($('sp-fit-status').textContent.includes('本轮最佳'));$('sp-train-one').click();await wait(()=>!state().busy);assert.equal(state().autoResult,null);
$('sp-auto-fit').click();await wait(()=>state().autoSession?.attempts>0);location.hash='construction';assert(!state().busy);assert.equal(state().autoResult.reason,'paused');assert($('sp-fit-status').textContent.includes('已暂停'));$('sp-reset').click();assert.equal(state().model.epoch,0);assert.equal(state().autoResult,null);
location.hash='kinetics';assert.equal(window.__latentLab.route,'kinetics');assert.equal(state().model.input,12);assert.equal(state().rows[0].time.length,6);assert.equal(state().model.epoch,0);
const kineticsToolbar=document.all().find(e=>e.attrs.class?.split(' ').includes('fit-toolbar'));
for(const id of ['kl-mode','kl-step','kl-train','kl-auto-fit','kl-pause','kl-reset']){assert(kineticsToolbar.all().includes($(id)),id+' must be in top toolbar');assert.equal(document.all().filter(e=>e.attrs.id===id).length,1);}
assert.equal(JSON.stringify(state().model.data),data);change('kl-cell',8);change('kl-gene',2);assert.equal(state().selected,8);assert.equal(state().gene,2);
assert.equal($('kl-comparison').querySelectorAll('tr').length,13);assert.equal($('kl-gradients').querySelectorAll('tr').length,11);
for(const button of document.querySelectorAll('[data-kl-stage]'))button.click();assert.equal(state().stage,5);
const meanK=JSON.stringify(state().rows[8].mu),varianceK=JSON.stringify(state().rows[8].variance),lossK=state().loss.loss;
$('kl-resample').click();assert.equal(state().representation,'sample');assert.equal(JSON.stringify(state().rows[8].mu),meanK);assert.equal(JSON.stringify(state().rows[8].variance),varianceK);assert.equal(state().loss.loss,lossK);
const sampleK=JSON.stringify(state().rows[8].z);$('kl-resample').click();assert.notEqual(JSON.stringify(state().rows[8].z),sampleK);
const allParams=JSON.stringify(state().model.p),timesK=JSON.stringify(state().rows[8].time);$('kl-cursor').value=0;$('kl-cursor').oninput();assert.equal(state().cursor,0);assert.equal(JSON.stringify(state().model.p),allParams);assert.equal(JSON.stringify(state().rows[8].time),timesK);
$('kl-follow').click();assert.equal(state().cursor,null);const yawK=state().yaw;$('kl-latent').onkeydown({key:'ArrowLeft',preventDefault(){}});assert.notEqual(state().yaw,yawK);
const timeBefore=JSON.stringify(state().rows[8].time);$('kl-beta').onpointerdown();$('kl-beta').value=1.2;$('kl-beta').oninput();assert(Math.abs(state().model.rates()[2].beta-1.2)<1e-10);assert.equal(JSON.stringify(state().rows[8].time),timeBefore);
const betaBefore=state().model.p.kbeta2;$('kl-time-bias').onpointerdown();$('kl-time-bias').value=.6;$('kl-time-bias').oninput();assert.equal(state().model.p.kbeta2,betaBefore);assert.notEqual(JSON.stringify(state().rows[8].time),timeBefore);
$('kl-reset').click();assert.equal(state().model.epoch,0);change('kl-repr','mean');change('kl-mode','rates');const fixedTime=JSON.stringify(state().rows[8].time);$('kl-step').click();assert.equal(JSON.stringify(state().rows[8].time),fixedTime);assert.equal(state().model.last.groups.encoder.updateNorm,0);assert(state().model.last.groups.beta.updateNorm>0);
change('kl-mode','time');const fixedDynamics=JSON.stringify(state().model.rates());$('kl-step').click();assert.equal(JSON.stringify(state().model.rates()),fixedDynamics);assert(state().model.last.groups.time.updateNorm>0);
change('kl-mode','joint');change('kl-epochs',10);const beforeJoint=state().loss.loss;$('kl-train').click();await wait(()=>state().model.epoch===12&&!state().busy);assert(state().loss.loss<beforeJoint);assert.equal(JSON.stringify(state().model.data),data);assert(state().rows.every(r=>r.prediction.every(Number.isFinite)&&r.time.every(v=>v>0&&v<4)));
assert(Object.values(state().model.last.groups).every(g=>g.gradientNorm>0&&g.updateNorm>0));
$('kl-play').click();await wait(()=>state().cursor>0);$('kl-play').click();const cursorK=state().cursor;await new Promise(r=>setTimeout(r,120));assert.equal(state().cursor,cursorK);
$('kl-train').click();await wait(()=>state().model.epoch>12);const kineticsModel=state().model;location.hash='space';const pausedK=kineticsModel.epoch;await new Promise(r=>setTimeout(r,150));assert.equal(kineticsModel.epoch,pausedK);assert.equal(state().model.epoch,0);
location.hash='kinetics';assert.equal(state().model.epoch,pausedK);listeners.resize();await new Promise(r=>setTimeout(r,120));$('kl-reset').click();assert.equal(state().model.epoch,0);
$('kl-auto-fit').click();await wait(()=>state().autoSession?.attempts>=2);$('kl-pause').click();assert(!state().busy);assert.equal(state().autoResult.reason,'paused');assert(state().autoResult.attempts>=2);assert(Math.abs(state().loss.loss-state().autoResult.bestLoss)<1e-10);assert($('kl-fit-status').textContent.includes('本轮最佳'));
$('kl-step').click();assert.equal(state().autoResult,null);$('kl-auto-fit').click();await wait(()=>state().autoSession?.attempts>=1);const autoK=state().model;location.hash='overview';const stoppedAutoK=autoK.epoch;await new Promise(r=>setTimeout(r,120));assert.equal(autoK.epoch,stoppedAutoK);location.hash='kinetics';assert.equal(state().autoResult.reason,'paused');$('kl-reset').click();assert.equal(state().autoResult,null);
change('kl-epochs',10000);assert.equal(state().trainingEpochs,10000);const dynamicsBefore={...state().model.p};document.querySelectorAll('[data-kl-tune]').find(b=>b.dataset.klTune==='beta').click();await wait(()=>state().model.epoch===10&&!state().busy);assert.equal(state().mode,'beta');
for(const k of Object.keys(dynamicsBefore))if(!k.startsWith('kbeta'))assert.equal(state().model.p[k],dynamicsBefore[k]);assert(Object.keys(dynamicsBefore).some(k=>k.startsWith('kbeta')&&state().model.p[k]!==dynamicsBefore[k]));assert.equal(state().model.trainingDraw,10);
location.hash='overview';change('sp-epochs',10000);assert.equal(state().trainingEpochs,10000);const directBefore={...state().model.p};document.querySelectorAll('[data-sp-tune]').find(b=>b.dataset.spTune==='decoder').click();await wait(()=>state().model.epoch===10&&!state().busy);assert.equal(state().trainingMode,'decoder');
for(const k of Object.keys(directBefore))if(!/^(d|o)/.test(k))assert.equal(state().model.p[k],directBefore[k]);assert(Object.keys(directBefore).some(k=>/^(d|o)/.test(k)&&state().model.p[k]!==directBefore[k]));
assert(document.all().some(e=>e.attrs.class?.split(' ').includes('sp-stacked')));assert.equal(state().model.trainingDraw,10);assert.equal(state().model.assess().samples,8);$('sp-reset').click();assert.equal(state().model.trainingDraw,0);
const result={passed:true,engine:'Node VM with DOM and canvas test doubles',scope:'Six routes; paper-toy six-gene observations with a shared direct-reconstruction model and an independent joint ODE model. No visual layout verification here.',checks:['linked cell selection','twelve-dimensional input and output','ellipsoid scope, contour and rotation preserve decoding','resampling preserves ellipsoid center and variance','input perturbation preserves observations','three-dimensional sampling','encoder steps and hidden units','100 real direct-reconstruction updates','shared state across original pages','legacy route aliases','resize handlers','reset','kinetics route and six stages','same 300 observations','six gene-specific times','kinetics sample preview preserves mean, variance and training loss','observation cursor does not change parameters or fitted times','manual beta changes preserve times','time bias preserves dynamics','rates-only and time-only freeze isolation','10 real joint ODE updates and all parameter groups change','finite twelve-dimensional predictions','reservoir play and pause','kinetics route changes stop training','model states persist independently']};
result.checks.push('unique training controls in top toolbar on all six routes','automatic best-model selection finishes on original model','auto result agrees with current objective','manual updates clear stale automatic results','pause and route changes restore best automatic checkpoint','kinetics automatic start and pause','10000-epoch option on both models','decoder module button freezes encoder and latent heads','beta module button freezes all other parameter groups','fresh training sampling counter','independent 8-sample scoring','stacked three-space markup');
mkdirSync(new URL('../.checks/',import.meta.url),{recursive:true});writeFileSync(new URL('../.checks/interaction-check.json',import.meta.url),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
