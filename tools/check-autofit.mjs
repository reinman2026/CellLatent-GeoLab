import assert from 'node:assert/strict';
import {NetworkLab,DynamicsLab,AutoFitSession} from '../dist/core.mjs';

const makeDirect=()=>new NetworkLab(3,false,{genes:3,cellCount:24,standardize:true});
const report=[];
for(const create of [makeDirect,()=>new DynamicsLab()]){
  const model=create(),data=JSON.stringify(model.data),initial=model.evaluate({lambda:.05});
  // A deterministic, actual parameter deterioration exercises best-state
  // restoration, including Adam continuation, rather than just loss labels.
  const session=new AutoFitSession(model,initial,{minSteps:3,patience:2,maxSteps:10,relativeTolerance:1e6});
  let bestCheckpoint;
  for(let i=0;i<3;i++){
    model.step({lambda:.05,lr:.008});
    if(i===2)model.p[model instanceof DynamicsLab?'kb0':'ob0']+=8;
    const metrics=model.evaluate({lambda:.05});session.observe(model,metrics);
    if(session.best.epoch===model.epoch)bestCheckpoint={p:{...model.p},m:{...model.m},v:{...model.v},optimizerStep:model.optimizerStep,parameterSteps:model.parameterSteps?{...model.parameterSteps}:undefined,trainingDraw:model.trainingDraw,epoch:model.epoch};
  }
  assert(session.done);assert.equal(session.reason,'plateau');assert(bestCheckpoint);assert(session.best.metrics.loss<initial.loss);
  const result=session.finish(model);assert.equal(result.attempts,3);assert(result.bestStep<result.attempts);assert(result.restored);assert.equal(model.epoch,3);
  assert.deepEqual(model.p,bestCheckpoint.p);assert.deepEqual(model.m,bestCheckpoint.m);assert.deepEqual(model.v,bestCheckpoint.v);
  assert(Math.abs(model.evaluate({lambda:.05}).loss-result.bestLoss)<1e-10);
  const reference=create();Object.assign(reference,bestCheckpoint);reference.history=[];
  model.step({lambda:.05,lr:.008});reference.step({lambda:.05,lr:.008});
  for(const key of Object.keys(model.p))assert(Math.abs(model.p[key]-reference.p[key])<1e-12,`Adam continuation mismatch ${key}`);
  assert.equal(JSON.stringify(model.data),data);
  const limited=new AutoFitSession(model,model.evaluate({lambda:.05}),{maxSteps:2});
  for(let i=0;i<2;i++){model.step({lambda:.05,lr:.008});limited.observe(model,model.evaluate({lambda:.05}));}
  assert.equal(limited.reason,'limit');assert.equal(limited.finish(model).attempts,2);
  const paused=new AutoFitSession(model,model.evaluate({lambda:.05}));assert.equal(paused.finish(model).reason,'paused');
  report.push({model:model.constructor.name,passed:true,bestStep:result.bestStep,attempts:result.attempts,bestLoss:result.bestLoss,checks:['plateau stops automatically','actual best weights restored','objective matches reported best','actual update count preserved','Adam continuation matches saved checkpoint','attempt limit','pause before first update','observations unchanged']});
}
console.log(JSON.stringify(report,null,2));
