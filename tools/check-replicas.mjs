// Synthetic TEST FIXTURES only. These values are never published as fit results.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {validateReplicas,replicaSelection} from '../dist/replica-lab.mjs';
import {stabilityReplicas} from '../dist/stability-replicas.mjs';
import {convexHull3D,minimumVolumeEnclosingEllipsoid3D} from '../dist/geometry3d.mjs';

for(const model of stabilityReplicas.models){
  const hull=convexHull3D(model.mu);assert(hull.faces.length>=4);assert(hull.edges.length>=6);
  for(const face of hull.faces){const [a,b,c]=face.map(i=>model.mu[i]),cross=(u,v)=>[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],normal=cross(b.map((v,j)=>v-a[j]),c.map((v,j)=>v-a[j])),sides=model.mu.map(point=>normal.reduce((sum,v,j)=>sum+v*(point[j]-a[j]),0));assert(sides.every(v=>v>=-1e-6)||sides.every(v=>v<=1e-6),'Hull face must support every cell mean.');}
  const mvee=minimumVolumeEnclosingEllipsoid3D(model.mu);assert(mvee.converged);assert(mvee.axes.every(v=>v>0));assert(mvee.maxScore<=1.01,`MVEE must enclose all means (max score ${mvee.maxScore}).`);
}

const pending=structuredClone(stabilityReplicas);
// Keep the pending fixture independent from the shipped real results.
pending.models=[];pending.status='awaiting-runtime';
assert.equal(pending.models.length,0,'Shipped results must remain pending until real training completes');
validateReplicas(pending);
const n=pending.data.cells.length,h=8,inputDim=pending.data.featureOrder.length,geneCount=pending.data.genes.length,expectedModels=pending.expectedModelCount;
const matrix=(rows,cols,f=()=>0)=>Array.from({length:rows},(_,i)=>Array.from({length:cols},(_,j)=>f(i,j)));
const fixture=structuredClone(pending);fixture.status='complete';fixture.blocker='TEST FIXTURE';
fixture.models=Array.from({length:expectedModels},(_,index)=>{
  const mu=matrix(n,3,(i,j)=>index+.01*i+j*.1),variance=matrix(n,3,(i,j)=>.1+.01*j),pass=k=>({prediction:matrix(n,inputDim,(i,j)=>.3+index+k*.1+i*.01+j*.02),time:matrix(n,geneCount,(i,j)=>1+k*.1+j*.2)});
  return {index,seed:[11,29,47][index],implementation:'complete-local-regvelo',observationSha256:'a'.repeat(64),epochs:5,bestEpoch:4,
    config:{n_latent:3,n_hidden:h,t_max:20},source:{model:'test-only',module:'test-only'},split:{train:Array.from({length:n-3},(_,i)=>i),validation:[n-3,n-2,n-1]},
    mu,variance,initial:{mu:matrix(n,3),variance:matrix(n,3,()=>1)},meanPass:pass(0),
    samples:[0,1,2].map(k=>{const epsilon=matrix(n,3,(i,j)=>(k-1)*.5+j*.1);return {epsilon,z:mu.map((p,i)=>p.map((v,j)=>v+Math.sqrt(variance[i][j])*epsilon[i][j])),...pass(k+1)};}),
    encoder:{hidden:matrix(n,h),input:pending.data.cells.map(c=>c.raw),meanWeight:matrix(3,h),meanBias:[0,0,0],varianceWeight:matrix(3,h),varianceBias:[0,0,0],varianceEpsilon:.0001},
    timeDecoder:{hiddenMu:matrix(n,h),weight:matrix(geneCount,h),bias:Array(geneCount).fill(0)},rates:{beta:Array.from({length:geneCount},(_,i)=>1+i),gamma:Array.from({length:geneCount},(_,i)=>.2+i*.1),W:matrix(geneCount,geneCount),b:Array(geneCount).fill(0),alphaObserved:matrix(n,geneCount,()=>1)},history:[{epoch:1,elbo_train:9,elbo_validation:10},{epoch:2,elbo_train:8,elbo_validation:9}]};
});
validateReplicas(fixture);
const before=replicaSelection(fixture,4,0),after=replicaSelection(fixture,4,1);
for(let i=0;i<3;i++){
  assert.deepEqual(before[i].observation,pending.data.cells[4].raw);
  assert.deepEqual(before[i].mu,after[i].mu);assert.deepEqual(before[i].variance,after[i].variance);
  assert.notDeepEqual(before[i].z,after[i].z);assert.notDeepEqual(before[i].prediction,after[i].prediction);
}
assert.deepEqual(replicaSelection(fixture,4,1,'mean')[0].z,fixture.models[0].mu[4]);
const missing=structuredClone(fixture);missing.models=[missing.models[1]];missing.status='partial';validateReplicas(missing);assert.equal(replicaSelection(missing,0,0)[0].model,null);
for(const mutate of [x=>x.models[1].observationSha256='b'.repeat(64),x=>x.models[1].variance[0][0]=-1,x=>x.models[1].samples[0].z[0][0]+=1,x=>x.models[1].split.validation=[0,1,2],x=>x.models[1].samples[0].epsilon[0][0]+=1,x=>x.models[1].implementation='browser-teaching',x=>x.data.cells[0].raw.pop()]){const invalid=structuredClone(fixture);mutate(invalid);assert.throws(()=>validateReplicas(invalid));}

const old=readFileSync(new URL('./check-interactions.mjs',import.meta.url),'utf8');
const scaffold=old.slice(old.indexOf('const ctx='),old.indexOf('const html=')).replace('this.disabled=false','this.disabled="disabled" in attrs');
const harness=vm.createContext({});vm.runInContext(scaffold+'\nglobalThis.ElementForTest=Element;',harness);
const Element=harness.ElementForTest;
Object.defineProperties(Element.prototype,{width:{get(){return Number(this.attrs.width??420);}},height:{get(){return Number(this.attrs.height??270);}}});
const html=readFileSync(new URL('../离线版.html',import.meta.url),'utf8');
const offlineScript=html.match(/<script>([\s\S]*?)<\/script>/)[1];
function boot(results=pending){
  const document=new Element('document');document.innerHTML=html.split('<script>')[0].replace(/<style>[\s\S]*?<\/style>/g,'');document.getElementById=id=>document.all().find(e=>e.attrs.id===id)??null;
  const listeners={},location={_hash:'',get hash(){return this._hash;},set hash(v){this._hash=v.startsWith('#')?v:'#'+v;listeners.hashchange?.();}};
  const window={document,location,devicePixelRatio:1,scrollTo(){},addEventListener:(name,cb)=>listeners[name]=cb};
  const context=vm.createContext({window,document,location,console,setTimeout,clearTimeout});
  let script=offlineScript;
  if(results){const paperFixture={format:'regvelo-paper-toy-stability-v1',status:results.status,generatedAt:'test',expectedModelCount:expectedModels,representatives:results.models.map((m,index)=>({...m,index})),data:results.data,provenance:{test:true}};const start=script.indexOf('const paperToyStability ='),marker=script.indexOf('// Surface of',start),end=script.lastIndexOf('};',marker);if(start<0||end<0)throw Error(`offline fixture anchors missing: ${start}/${marker}/${end}`);script=script.slice(0,start)+`const paperToyStability = ${JSON.stringify(paperFixture)};`+script.slice(end+2);}
  if(!script.includes('function createSpatialLab'))throw Error(`offline fixture injection removed spatial lab; len=${script.length}, paper=${script.indexOf('paperToyStability')}, replica=${script.indexOf('stabilityReplicas')}`);
  vm.runInContext(script,context,{timeout:10000});location.hash='replica-compare';
  return {document,location,window,$:id=>{const node=document.getElementById(id);assert(node,`Missing ${id}`);return node;},state:()=>window.__latentLab.state()};
}
const p=boot();assert.equal(p.window.__latentLab.route,'replica-compare');assert.equal(p.document.querySelectorAll('canvas').length,15);assert(p.$('rp-train').disabled);assert.equal(p.state().bundle.models.length,0);assert.equal(p.state().explanationModel,null);
const a=boot(fixture);assert.equal(a.document.querySelectorAll('canvas').length,15);a.$('rp-cell').value=4;a.$('rp-cell').onchange();assert.equal(a.state().selected,4);
const frozen=JSON.stringify(a.state().rows.map(r=>[r.mu,r.variance,r.model.rates]));const previous=JSON.stringify(a.state().rows.map(r=>r.z));a.$('rp-sample').click();assert.notEqual(JSON.stringify(a.state().rows.map(r=>r.z)),previous);assert.equal(JSON.stringify(a.state().rows.map(r=>[r.mu,r.variance,r.model.rates])),frozen);
const yaw=a.state().viewAngles[2].yaw;a.$('rp-latent-2').onkeydown({key:'ArrowLeft',preventDefault(){}});assert.notEqual(a.state().viewAngles[2].yaw,yaw);assert.equal(a.state().explanationModel.index,0);
assert.equal(a.state().orderedCells.length,n);assert.equal(new Set(a.state().orderedCells).size,n);assert.equal(a.state().showGeometry,true);assert.equal(a.state().showEllipsoids,true);assert.equal(a.state().showHull,false);assert.equal(a.state().showMvee,false);assert.equal(a.state().showPoints,true);for(const [id,key] of [['rp-geometry','showGeometry'],['rp-ellipsoids','showEllipsoids'],['rp-hull','showHull'],['rp-mvee','showMvee'],['rp-points','showPoints']]){a.$(id).checked=true;a.$(id).onchange();assert.equal(a.state()[key],true);a.$(id).checked=false;a.$(id).onchange();assert.equal(a.state()[key],false);}a.$('rp-flow').click();assert.equal(a.state().showFlow,true);a.$('rp-order').value='sample';a.$('rp-order').onchange();assert.equal(Array.from(a.state().orderedCells).join(','),Array.from({length:n},(_,i)=>i).join(','));a.$('rp-flow').click();assert.equal(a.state().showFlow,false);
for(const route of ['replica-encoder','replica-training','replica-space','replica-kinetics','replica-source']){a.location.hash=route;assert.equal(a.window.__latentLab.route,route);assert.equal(a.state().explanationModel.index,0);assert(a.document.querySelectorAll('[data-rp-scope]').every(b=>b.disabled));}
const b=boot(pending);b.location.hash='replica-kinetics';assert.equal(b.state().explanationModel,null);assert(b.$('rp-detail').textContent.includes('尚未训练完成'));
const keep=JSON.stringify(a.state().bundle);a.$('rp-import').files=[{size:5,text:async()=>'{bad'}];await a.$('rp-import').onchange();assert.equal(JSON.stringify(a.state().bundle),keep);assert(a.$('rp-status').textContent.includes('导入失败'));
console.log('Replica checks passed: pending honesty, 15 plots, same observations, fixed μ/v on resampling, exact sample outputs, model-1 explanations, malformed input rejection, route handlers. No actual fits or visual browser QA performed.');
