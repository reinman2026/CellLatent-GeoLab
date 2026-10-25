// A small, explicitly educational VAE + two-gene differentiable ODE.
// No RegVelo data or fitted parameters are embedded in this file.
export class RNG {
  constructor(seed=42){this.s=seed>>>0;}
  uniform(){this.s=(1664525*this.s+1013904223)>>>0;return (this.s+.5)/4294967296;}
  normal(){return Math.sqrt(-2*Math.log(this.uniform()))*Math.cos(2*Math.PI*this.uniform());}
}
export const softplus=x=>x>25?x:Math.log1p(Math.exp(x));
export const sigmoid=x=>1/(1+Math.exp(-Math.max(-40,Math.min(40,x))));
export function gaussian(x,mu,sd){return Math.exp(-.5*((x-mu)/sd)**2)/(Math.sqrt(2*Math.PI)*sd);}
export function analyticPosterior(y,a,tau){const v=tau*tau/(tau*tau+a*a);return {mu:a*y/(tau*tau+a*a),variance:v,sd:Math.sqrt(v)};}
export function klGaussian(mu,v,pm,pv){return .5*((v+(mu-pm)**2)/pv-1+Math.log(pv/v));}
export function sampleNormal(mu,variance,rng){return mu.map((m,j)=>m+Math.sqrt(variance[j])*rng.normal());}
export function demonstrationCells(d=10){
  return Array.from({length:5},(_,i)=>({name:`细胞 ${i+1}`,mu:Array.from({length:d},(_,j)=>Math.sin((i+1)*(j+1)*.72)*1.35+(i-2)*.12),variance:Array.from({length:d},(_,j)=>.12+.12*((i+j)%4))}));
}

let tape=[];
class A {
  constructor(v,back=null){this.v=Number(v);this.g=0;this.back=back;tape.push(this);}
}
const av=x=>x instanceof A?x:new A(x);
function add(a,b){a=av(a);b=av(b);const n=new A(a.v+b.v,()=>{a.g+=n.g;b.g+=n.g;});return n;}
function sub(a,b){a=av(a);b=av(b);const n=new A(a.v-b.v,()=>{a.g+=n.g;b.g-=n.g;});return n;}
function mul(a,b){a=av(a);b=av(b);const n=new A(a.v*b.v,()=>{a.g+=n.g*b.v;b.g+=n.g*a.v;});return n;}
function div(a,b){a=av(a);b=av(b);const n=new A(a.v/b.v,()=>{a.g+=n.g/b.v;b.g-=n.g*a.v/(b.v*b.v);});return n;}
function log(a){a=av(a);const n=new A(Math.log(a.v),()=>{a.g+=n.g/a.v;});return n;}
function sqrt(a){a=av(a);const n=new A(Math.sqrt(a.v),()=>{a.g+=n.g/(2*n.v);});return n;}
function sp(a){a=av(a);const n=new A(softplus(a.v),()=>{a.g+=n.g*sigmoid(a.v);});return n;}
function sig(a){a=av(a);const n=new A(sigmoid(a.v),()=>{a.g+=n.g*n.v*(1-n.v);});return n;}
const sum=xs=>xs.reduce((a,b)=>add(a,b),av(0));
const sq=a=>mul(a,a);
function backward(loss){loss.g=1;for(let i=tape.length-1;i>=0;i--)if(tape[i].g&&tape[i].back)tape[i].back();}
const invsp=x=>Math.log(Math.expm1(x));

function derivative(y,p){
  const alpha=[0,1].map(g=>sp(add(p[`b${g}`],sum([0,1].map(j=>mul(p[`w${g}${j}`],y[2+j]))))));
  const beta=[0,1].map(g=>add(sp(p[`beta${g}`]),.1));
  const gamma=[0,1].map(g=>add(sp(p[`gamma${g}`]),.1));
  return [sub(alpha[0],mul(beta[0],y[0])),sub(alpha[1],mul(beta[1],y[1])),sub(mul(beta[0],y[0]),mul(gamma[0],y[2])),sub(mul(beta[1],y[1]),mul(gamma[1],y[3]))];
}
function integrate(t,p,steps=6){
  let y=[0,0,0,0].map(av);const dt=div(t,steps);
  const shifted=(state,k,f)=>state.map((v,j)=>add(v,mul(mul(dt,f),k[j])));
  for(let k=0;k<steps;k++){
    const k1=derivative(y,p),k2=derivative(shifted(y,k1,.5),p),k3=derivative(shifted(y,k2,.5),p),k4=derivative(shifted(y,k3,1),p);
    y=y.map((v,j)=>add(v,mul(div(dt,6),sum([k1[j],mul(2,k2[j]),mul(2,k3[j]),k4[j]]))));
  }
  return y;
}
function numericState(t,W,b,beta,gamma,steps=160){
  let y=[0,0,0,0];const dt=t/steps;
  const f=s=>{const a=b.map((v,g)=>softplus(v+W[g][0]*s[2]+W[g][1]*s[3]));return [a[0]-beta[0]*s[0],a[1]-beta[1]*s[1],beta[0]*s[0]-gamma[0]*s[2],beta[1]*s[1]-gamma[1]*s[3]];};
  const shift=(s,k,c)=>s.map((v,j)=>v+dt*c*k[j]);
  for(let i=0;i<steps;i++){const a=f(y),b1=f(shift(y,a,.5)),c=f(shift(y,b1,.5)),d=f(shift(y,c,1));y=y.map((v,j)=>v+dt*(a[j]+2*b1[j]+2*c[j]+d[j])/6);}
  return y;
}
export function makeData(){
  return Array.from({length:10},(_,i)=>{const time=.25+i*.35;const y=numericState(time,[[.12,.5],[.4,-.1]],[.3,.1],[.7,.9],[.45,.55]);return {name:`细胞 ${i+1}`,x:y,referenceTime:time};});
}

export class ToyVAE {
  constructor(seed=42){this.seed=seed;this.data=makeData();this.reset();}
  reset(){
    const r=new RNG(this.seed);this.p={};
    for(let d=0;d<2;d++){
      this.p[`mb${d}`]=0;this.p[`vb${d}`]=-2;
      for(let j=0;j<4;j++){this.p[`m${d}${j}`]=r.normal()*.09;this.p[`v${d}${j}`]=r.normal()*.012;}
    }
    for(let g=0;g<2;g++){
      this.p[`tb${g}`]=-.2+g*.25;
      for(let d=0;d<2;d++)this.p[`t${g}${d}`]=r.normal()*.25;
      for(let j=0;j<2;j++)this.p[`w${g}${j}`]=r.normal()*.04;
      this.p[`b${g}`]=.15;this.p[`beta${g}`]=invsp(.65+g*.1);this.p[`gamma${g}`]=invsp(.4+g*.1);
    }
    this.m={};this.v={};this.epoch=0;this.history=[];this.last=null;
  }
  encoder(i){
    const row=this.data[i].x;const x=[row[2],row[3],row[0],row[1]].map(v=>v/2);
    const mu=[0,1].map(d=>this.p[`mb${d}`]+x.reduce((s,x,j)=>s+x*this.p[`m${d}${j}`],0));
    const variance=[0,1].map(d=>softplus(this.p[`vb${d}`]+x.reduce((s,x,j)=>s+x*this.p[`v${d}${j}`],0))+1e-4);
    return {mu,variance};
  }
  noise(epoch=0){const r=new RNG(this.seed+991+epoch*137);return this.data.map(()=>[r.normal(),r.normal()]);}
  evaluate({lambda=.05,fixedNoise=true,gradient=false,selected=0,epsilon=null,computeLoss=true}={}){
    tape=[];const p=Object.fromEntries(Object.entries(this.p).map(([k,v])=>[k,av(v)]));
    const noise=epsilon??this.noise(fixedNoise?0:this.epoch);let totalRec=av(0),totalKL=av(0);let selectedSnapshot=null;const cells=[];
    this.data.forEach((row,i)=>{
      const x=[row.x[2],row.x[3],row.x[0],row.x[1]].map(v=>v/2);
      const mu=[0,1].map(d=>add(p[`mb${d}`],sum(x.map((v,j)=>mul(v,p[`m${d}${j}`])))));
      const variance=[0,1].map(d=>add(sp(add(p[`vb${d}`],sum(x.map((v,j)=>mul(v,p[`v${d}${j}`]))))),1e-4));
      const z=mu.map((m,d)=>add(m,mul(sqrt(variance[d]),noise[i][d])));
      const time=[0,1].map(g=>mul(4,sig(add(p[`tb${g}`],sum(z.map((v,d)=>mul(v,p[`t${g}${d}`])))))));
      const states=time.map(t=>integrate(t,p));
      const pred=[states[0][0],states[1][1],states[0][2],states[1][3]];
      const rec=computeLoss?sum(pred.map((v,j)=>add(div(sq(sub(row.x[j],v)),2*.18*.18),Math.log(.18)+.5*Math.log(2*Math.PI)))):null;
      const kl=computeLoss?mul(.5,sum(mu.map((m,d)=>sub(sub(add(sq(m),variance[d]),log(variance[d])),1)))):null;
      if(computeLoss){totalRec=add(totalRec,rec);totalKL=add(totalKL,kl);}
      const c={mu:mu.map(v=>v.v),variance:variance.map(v=>v.v),z:z.map(v=>v.v),time:time.map(v=>v.v),pred:pred.map(v=>v.v)};cells.push(c);
      if(i===selected)selectedSnapshot={...c,eps:noise[i],observed:row.x.slice(),reconstruction:rec?.v??null,kl:kl?.v??null};
    });
    if(!computeLoss){tape=[];return {loss:null,reconstruction:null,kl:null,snapshot:selectedSnapshot,cells,gradients:null};}
    const rec=div(totalRec,this.data.length),kl=div(totalKL,this.data.length),loss=add(rec,mul(lambda,kl));
    if(gradient)backward(loss);
    const result={loss:loss.v,reconstruction:rec.v,kl:kl.v,snapshot:selectedSnapshot,cells,gradients:gradient?Object.fromEntries(Object.entries(p).map(([k,v])=>[k,v.g])):null};
    tape=[];return result;
  }
  step(options={}){
    const r=this.evaluate({...options,gradient:true});
    if(!Number.isFinite(r.loss)||Object.values(r.gradients).some(v=>!Number.isFinite(v)))return {...r,error:'数值无效，请重置或减小学习率'};
    const norm=Math.sqrt(Object.values(r.gradients).reduce((s,v)=>s+v*v,0));const clip=Math.min(1,5/(norm+1e-12));
    const lr=options.lr??.012;const n=this.epoch+1;const changes={};
    for(const k of Object.keys(this.p)){
      const before=this.p[k],g=r.gradients[k]*clip;this.m[k]=.9*(this.m[k]??0)+.1*g;this.v[k]=.999*(this.v[k]??0)+.001*g*g;
      this.p[k]-=lr*(this.m[k]/(1-.9**n))/(Math.sqrt(this.v[k]/(1-.999**n))+1e-8);
      changes[k]={before,gradient:r.gradients[k],after:this.p[k]};
    }
    this.epoch=n;this.last={...r,changes,norm};this.history.push({epoch:n,loss:r.loss,reconstruction:r.reconstruction,kl:r.kl});return this.last;
  }
  infer(options={}){return this.evaluate({...options,gradient:false,computeLoss:false});}
}

export function gradientCheck(){
  const model=new ToyVAE(42),r=model.evaluate({gradient:true});const checks=[];
  for(const key of ['m00','v00','t00','w00','beta0','gamma0']){
    const old=model.p[key],h=1e-5;model.p[key]=old+h;const plus=model.evaluate().loss;model.p[key]=old-h;const minus=model.evaluate().loss;model.p[key]=old;
    const numerical=(plus-minus)/(2*h),analytic=r.gradients[key],relativeError=Math.abs(numerical-analytic)/Math.max(1,Math.abs(numerical),Math.abs(analytic));checks.push({key,numerical,analytic,relativeError});
  }
  return checks;
}
