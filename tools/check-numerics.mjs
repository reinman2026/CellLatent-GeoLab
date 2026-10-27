import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import vm from 'node:vm';
import {NetworkLab,gradientCheck,ToyVAE,encoderTrace,twoPoolVelocity} from '../dist/core.mjs';
import {ellipsoidPoint,projectEllipsoid,orientLatentProjection,rotateLatent,umapEmbedding} from '../dist/spatial-lab.mjs';

const report={gradients:[],training:[],checks:[]};
const spatial=new NetworkLab(3,false,{genes:3,cellCount:24,standardize:true});
assert.equal(spatial.kinetics.length,3);assert(spatial.kinetics.every(p=>p.alpha>0&&p.beta>0&&p.gamma>0&&p.fit.samples>0));
assert.equal(twoPoolVelocity(spatial.data[0].raw,spatial.kinetics).length,6);
const umap=umapEmbedding(spatial.data.map((_,i)=>spatial.trace(i,{useMean:true}).mu));assert.equal(umap.length,24);assert(umap.every(p=>p.length===2&&p.every(Number.isFinite)));assert(new Set(umap.map(p=>p.join(','))).size>3);
const gradients=spatial.evaluate({lambda:.05,gradient:true}).gradients;
for(const key of ['h0_0','mu0_0','va0_0','d0_0','o0_0']){
  const value=spatial.p[key],eps=1e-5;spatial.p[key]=value+eps;const a=spatial.evaluate({lambda:.05}).loss;spatial.p[key]=value-eps;const b=spatial.evaluate({lambda:.05}).loss;spatial.p[key]=value;
  assert(Math.abs((a-b)/(2*eps)-gradients[key])<1e-6);
}
for(let j=0;j<6;j++)assert(Math.abs(spatial.data.reduce((s,r)=>s+spatial.inputVector(r.raw)[j],0))<1e-10);
for(let i=0;i<24;i++){
  const t=spatial.trace(i,{useMean:true});assert.equal(t.raw.length,6);assert.equal(t.mu.length,3);assert.equal(t.prediction.length,6);
  assert.deepEqual(t.prediction,spatial.decode(t.mu).prediction);
  for(const k of [1,2,2.795483482915]){
    const p=ellipsoidPoint(t.mu,t.variance,k,.41,-.73);
    assert(Math.abs(p.reduce((s,v,j)=>s+(v-t.mu[j])**2/t.variance[j],0)-k*k)<1e-11);
  }
}
const orientation=projectEllipsoid([1,2,3],[.3,1.7,.8],1,.5,.25),frame=orientLatentProjection([1,2,3],[.3,1.7,.8],.5,.25),major=frame.rotate([orientation.radii[0]*Math.cos(orientation.angle),orientation.radii[0]*Math.sin(orientation.angle)]),minor=frame.rotate([-orientation.radii[1]*Math.sin(orientation.angle),orientation.radii[1]*Math.cos(orientation.angle)]);
assert(Math.abs(major[1])<1e-10&&Math.abs(Math.abs(major[0])-orientation.radii[0])<1e-10);assert(Math.abs(minor[0])<1e-10&&Math.abs(Math.abs(minor[1])-orientation.radii[1])<1e-10);
const close=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const mean=[1,-2,3],variance=[4,1,9];
assert.deepEqual(ellipsoidPoint(mean,variance,1,0,0),[3,-2,3]);
projectEllipsoid(mean,variance,1,0,0).radii.forEach((v,j)=>close(v,[2,1][j]));
projectEllipsoid(mean,variance,1,Math.PI/2,0).radii.forEach((v,j)=>close(v,[3,1][j]));
for(const [yaw,pitch] of [[0,0],[.6,-.4],[1.2,.7]]){
  const k=2,shape=projectEllipsoid(mean,variance,k,yaw,pitch);
  assert.deepEqual(shape.center,rotateLatent(mean,yaw,pitch));
  // Independently reconstruct the projected covariance from ellipse axes.
  const [a,b]=shape.radii.map(v=>v*v),c=Math.cos(shape.angle),s=Math.sin(shape.angle);
  close(shape.covariance[0][0],a*c*c+b*s*s);
  close(shape.covariance[1][1],a*s*s+b*c*c);
  close(shape.covariance[0][1],(a-b)*c*s);
  for(let n=0;n<100;n++){
    const point=rotateLatent(ellipsoidPoint(mean,variance,k,n*.41,n*.73),yaw,pitch);
    const dx=point[0]-shape.center[0],dy=-(point[1]-shape.center[1]);
    assert((dx*c+dy*s)**2/a+(-dx*s+dy*c)**2/b<=1+1e-12);
  }
}
const startSpatial=spatial.evaluate({lambda:.05}).loss;for(let i=0;i<100;i++)spatial.step({lambda:.05,lr:.008});assert(spatial.evaluate({lambda:.05}).loss<startSpatial);
report.spatial={shape:[6,3,6],start:startSpatial,end:spatial.evaluate({lambda:.05}).loss,steps:100,standardization:true,ellipsoidSurfaceAndProjection:true,twoPoolKinetics:true};
for(const dim of [2,10])for(const logInput of [false,true]){
  const net=new NetworkLab(dim,logInput),p=net.evaluate({gradient:true});
  // Check a meaningful nonzero parameter from every trainable layer.
  for(const prefix of ['h','mu','va','d','o']){
    const key=Object.keys(p.gradients).filter(k=>k.startsWith(prefix)).sort((a,b)=>Math.abs(p.gradients[b])-Math.abs(p.gradients[a]))[0];
    const old=net.p[key],eps=1e-5;net.p[key]=old+eps;const plus=net.evaluate().loss;net.p[key]=old-eps;const minus=net.evaluate().loss;net.p[key]=old;
    const numeric=(plus-minus)/(2*eps),analytic=p.gradients[key],error=Math.abs(numeric-analytic)/Math.max(1,Math.abs(numeric),Math.abs(analytic));
    assert(error<1e-5,`${dim}/${logInput}/${key}: gradient error ${error}`);report.gradients.push({dim,logInput,key,error});
  }
  // Independently reconstruct the likelihood and KL from numerical traces.
  let rec=0,kl=0;
  for(let i=0;i<net.data.length;i++){
    const t=net.trace(i);assert.equal(t.mu.length,dim);assert(t.variance.every(v=>v>0));
    rec+=t.prediction.reduce((s,v,j)=>s+(v-net.data[i].raw[j])**2/(2*.3**2)+Math.log(.3)+.5*Math.log(2*Math.PI),0)/net.data.length;
    kl+=t.mu.reduce((s,v,j)=>s+.5*(v*v+t.variance[j]-Math.log(t.variance[j])-1),0)/net.data.length;
  }
  assert(Math.abs(rec-p.reconstruction)<1e-9);assert(Math.abs(kl-p.kl)<1e-9);
  const original=JSON.stringify(net.data),a=net.trace(0),b=net.trace(0,{serial:1});
  assert.deepEqual(a.mu,b.mu);assert.deepEqual(a.variance,b.variance);assert.notDeepEqual(a.z,b.z);
  assert.notDeepEqual(a.mu,net.trace(0,{gene:0,delta:1}).mu);assert.equal(JSON.stringify(net.data),original);
  const start=net.evaluate().loss;for(let i=0;i<80;i++)net.step();const end=net.evaluate().loss;
  assert(Number.isFinite(end)&&end<start,`Training failed to improve ${dim}/${logInput}`);assert.equal(JSON.stringify(net.data),original);report.training.push({dim,logInput,start,end,steps:80});
}
const linear=new ToyVAE();for(let i=0;i<10;i++)assert.deepEqual(encoderTrace(linear,i).mu.value,linear.encoder(i).mu);
assert(gradientCheck().every(x=>x.relativeError<1e-5));
const html=readFileSync(new URL('../离线版.html',import.meta.url),'utf8'),script=html.match(/<script>([\s\S]*?)<\/script>/)[1];new vm.Script(script);
assert(!/^\s*(import|export) /m.test(script));assert(!html.includes('\\boxed'));
report.checks.push('independent likelihood and KL agree with AD','all input transforms and latent sizes','resampling preserves distribution','perturbation leaves observations unchanged','old encoder and ODE gradients','offline classic-script parsing');
mkdirSync(new URL('../.checks/',import.meta.url),{recursive:true});writeFileSync(new URL('../.checks/numerical-check.json',import.meta.url),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
