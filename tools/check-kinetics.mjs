import assert from 'node:assert/strict';
import {DynamicsLab,NetworkLab,kineticState,kineticField} from '../dist/core.mjs';

const model=new DynamicsLab(),report={gradients:[],modes:[]};
assert.deepEqual(model.data,new NetworkLab(3,false,{genes:3,cellCount:24,standardize:true}).data);
const result=model.evaluate({gradient:true});
assert(Math.abs(result.loss-model.evaluate().loss)<1e-10);
for(const key of ['h0_0','mu0_0','va0_0','d0_0','tm0_0','tmb0','kw0_1','kb0','kbeta0','kgamma0']){
  const v=model.p[key],eps=1e-5;model.p[key]=v+eps;const a=model.evaluate().loss;model.p[key]=v-eps;const b=model.evaluate().loss;model.p[key]=v;
  const numerical=(a-b)/(2*eps),analytic=result.gradients[key],error=Math.abs(numerical-analytic)/Math.max(1,Math.abs(numerical),Math.abs(analytic));
  assert(error<2e-6,`${key}: ${error}`);report.gradients.push({key,error});
}
const state=kineticState(2,model.p),field=kineticField(state,model.p),rates=model.rates();
const alpha=rates.map((r,g)=>Math.log1p(Math.exp(r.bias+r.weights.reduce((s,w,j)=>s+w*state[3+j],0))));
assert(Math.abs(field.reduce((s,v)=>s+v,0)-alpha.reduce((s,v,g)=>s+v-rates[g].gamma*state[3+g],0))<1e-12);
const finer=kineticState(4,model.p,120),coarser=kineticState(4,model.p,12);
assert(Math.max(...finer.map((v,i)=>Math.abs(v-coarser[i])))<1e-4);
// Independent constant-alpha analytic solution, beta != gamma.
const p={...model.p};for(const k of Object.keys(p))if(k.startsWith('kw'))p[k]=0;
const y=kineticState(2,p,100);
for(let g=0;g<3;g++){
  const a=Math.log1p(Math.exp(p['kb'+g])),b=rates[g].beta,c=rates[g].gamma,t=2;
  const u=a/b*(1-Math.exp(-b*t)),s=a*((1-Math.exp(-c*t))/c-(Math.exp(-b*t)-Math.exp(-c*t))/(c-b));
  assert(Math.abs(y[g]-u)<1e-8);assert(Math.abs(y[g+3]-s)<1e-8);
}
// Time derivatives of the integrated solution agree with the ODE field.
const eps=1e-5,plus=kineticState(2+eps,model.p,100),minus=kineticState(2-eps,model.p,100),exact=kineticField(kineticState(2,model.p,100),model.p);
assert(Math.max(...exact.map((v,j)=>Math.abs(v-(plus[j]-minus[j])/(2*eps))))<1e-7);
for(const mode of ['rates','time','joint']){
  const m=new DynamicsLab(),before={...m.p},trace=m.trace(5,{useMean:true}),curve=kineticState(2,m.p);m.step({mode});
  const changed=[...new Set(Object.keys(m.p).filter(k=>m.p[k]!==before[k]).map(k=>m.parameterGroup(k)))].sort();
  assert.deepEqual(changed,mode==='rates'?['beta','gamma','grn']:mode==='time'?['time']:['beta','encoder','gamma','grn','time']);
  if(mode==='rates'){assert.deepEqual(m.trace(5,{useMean:true}).time,trace.time);assert.notDeepEqual(kineticState(2,m.p),curve);}
  if(mode==='time'){assert.deepEqual(kineticState(2,m.p),curve);assert.notDeepEqual(m.trace(5,{useMean:true}).time,trace.time);}
  report.modes.push({mode,changed});
}
const start=model.evaluate().loss;for(let i=0;i<30;i++)model.step();const end=model.evaluate().loss;assert(end<start*.7);
for(let i=0;i<24;i++){const t=model.trace(i);assert(t.time.every(v=>v>0&&v<4));assert(t.prediction.every(Number.isFinite));assert.equal(t.mu.length,3);assert.equal(t.prediction.length,6);}
assert(model.rates().every(r=>r.beta>0&&r.gamma>0));
report.training={steps:30,start,end};report.passed=true;
console.log(JSON.stringify(report,null,2));
