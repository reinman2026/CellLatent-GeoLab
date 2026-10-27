import assert from 'node:assert/strict';
import {NetworkLab,DynamicsLab} from '../dist/core.mjs';

const direct=()=>new NetworkLab(3,false,{genes:3,cellCount:24,standardize:true}),report=[];
for(const create of [direct,()=>new DynamicsLab()]){
  const m=create(),preview=JSON.stringify(m.trace(0).eps),bank=JSON.stringify(m.evaluationNoise),data=JSON.stringify(m.data),initial=m.assess();
  assert.deepEqual(m.assess(),initial);
  const samples=m.evaluationNoise.map(noise=>m.evaluate({lambda:.05,noise}));
  assert(Math.abs(initial.loss-samples.reduce((s,r)=>s+r.loss,0)/8)<1e-12);
  const noise=m.evaluationNoise[3],numeric=m.evaluate({lambda:.05,noise}),ad=m.evaluate({lambda:.05,noise,gradient:true});assert(Math.abs(numeric.loss-ad.loss)<1e-10);
  const key=m instanceof DynamicsLab?'tm0_0':'o0_0',old=m.p[key],eps=1e-5;m.p[key]=old+eps;const plus=m.evaluate({lambda:.05,noise}).loss;m.p[key]=old-eps;const minus=m.evaluate({lambda:.05,noise}).loss;m.p[key]=old;
  assert(Math.abs((plus-minus)/(2*eps)-ad.gradients[key])<1e-6);
  m.step({lambda:.05,lr:.008});const first=m.lastTrainingNoise.map(r=>r.slice());m.step({lambda:.05,lr:.008});
  assert.equal(m.trainingDraw,2);assert(m.lastTrainingNoise.every((row,i)=>row.every((v,j)=>v!==first[i][j])));
  assert.equal(JSON.stringify(m.trace(0).eps),preview);assert.equal(JSON.stringify(m.evaluationNoise),bank);assert.equal(JSON.stringify(m.data),data);
  const modes=m instanceof DynamicsLab?['encoder','grn','beta','gamma','time']:['encoder','latent','decoder'];
  for(const mode of modes){
    const model=create(),before={...model.p},trace=model.trace(7,{useMean:true});model.step({lambda:.05,lr:.008,mode});
    const changed=Object.keys(model.p).filter(k=>before[k]!==model.p[k]);assert(changed.length>0);
    assert(changed.every(k=>mode==='encoder'&&!(model instanceof DynamicsLab)?model.parameterGroup(k)!=='decoder':model.parameterGroup(k)===mode),mode);
    if(mode==='decoder'){assert.deepEqual(model.trace(7,{useMean:true}).mu,trace.mu);assert.deepEqual(model.trace(7,{useMean:true}).variance,trace.variance);}
    if(model instanceof DynamicsLab&&['grn','beta','gamma'].includes(mode))assert.deepEqual(model.trace(7,{useMean:true}).time,trace.time);
  }
  report.push({model:m.constructor.name,passed:true,checks:['fresh epsilon for every cell and dimension each update','preview epsilon stays independent','fixed 8-sample scoring matches explicit average','scoring is deterministic at fixed weights','provided-noise numeric and AD losses agree','provided-noise finite difference gradient','all module scopes update only their parameters','observations unchanged']});
}
console.log(JSON.stringify(report,null,2));
