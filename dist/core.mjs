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

// Fit the two-pool kinetic terms used by RegVelo from time-ordered S/U pairs.
// The teaching data has three interleaved groups; each group is treated as a
// short trajectory so the finite differences do not mix unrelated branches.
export function fitTwoPoolKinetics(data,{genes=3}={}){
  const groups=[...new Set(data.map((row,i)=>row.group??i%3))],series=groups.map(group=>data.map((row,i)=>({row,i})).filter(x=>(x.row.group??x.i%3)===group).sort((a,b)=>a.i-b.i));
  const regression=(xs,ys)=>{const mx=xs.reduce((s,v)=>s+v,0)/Math.max(1,xs.length),my=ys.reduce((s,v)=>s+v,0)/Math.max(1,ys.length),den=xs.reduce((s,v)=>s+(v-mx)**2,0);const slope=den>1e-9?xs.reduce((s,v,i)=>s+(v-mx)*(ys[i]-my),0)/den:0;return {slope,intercept:my-slope*mx};};
  const solve=(rows,ys)=>{const p=rows[0]?.length??0,A=Array.from({length:p},(_,i)=>Array.from({length:p+1},(_,j)=>j===p?0:0));for(let i=0;i<p;i++)for(let j=0;j<p;j++)A[i][j]=rows.reduce((s,r)=>s+r[i]*r[j],0)+(i===j?1e-5:0);for(let i=0;i<p;i++)A[i][p]=rows.reduce((s,r,k)=>s+r[i]*ys[k],0);for(let i=0;i<p;i++){let pivot=i;for(let k=i+1;k<p;k++)if(Math.abs(A[k][i])>Math.abs(A[pivot][i]))pivot=k;[A[i],A[pivot]]=[A[pivot],A[i]];const d=A[i][i]||1e-8;for(let j=i;j<=p;j++)A[i][j]/=d;for(let k=0;k<p;k++)if(k!==i){const q=A[k][i];for(let j=i;j<=p;j++)A[k][j]-=q*A[i][j];}}return A.map(r=>r[p]);};
  return Array.from({length:genes},(_,g)=>{
    const u=[],du=[],s=[],ds=[],regX=[];
    for(const path of series)for(let k=0;k<path.length-1;k++){
      const a=path[k].row.raw,b=path[k+1].row.raw,dt=1/Math.max(1,path.length-1);
      s.push(a[g]);u.push(a[genes+g]);ds.push((b[g]-a[g])/dt);du.push((b[genes+g]-a[genes+g])/dt);regX.push([1,...a.slice(0,genes),a[genes+g]]);
    }
    const reg=solve(regX,du),beta=Math.max(.03,-reg.at(-1)),alphaWeights=reg.slice(1,1+genes),alpha=Math.max(.01,reg[0]);
    const residual=ds.map((v,i)=>v-beta*u[i]),sr=regression(s,residual),gamma=Math.max(.03,-sr.slope);
    return {alpha,beta,gamma,alphaWeights,fit:{alphaIntercept:reg[0],gammaSlope:sr.slope,samples:u.length}};
  });
}

export function twoPoolVelocity(raw,kinetics,{genes=3}={}){
  const ds=[],du=[];for(let g=0;g<genes;g++){const p=kinetics[g],s=raw[g],u=raw[genes+g],alpha=Math.max(.01,p.alphaWeights?p.fit.alphaIntercept+p.alphaWeights.reduce((sum,w,j)=>sum+w*raw[j],0):p.alpha);du.push(alpha-p.beta*u);ds.push(p.beta*u-p.gamma*s);}
  return [...ds,...du];
}
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

// --- Encoder introspection: the exact numbers of one forward pass. ---------
// Everything below is derived from the current model.p, so the page prints the
// arithmetic the network actually performs, not a redrawn illustration.
export function encoderTrace(model,i){
  const row=model.data[i],p=model.p;
  const raw=[row.x[2],row.x[3],row.x[0],row.x[1]]; // spliced A, B then unspliced A, B
  const x=raw.map(v=>v/2);
  const head=(bKey,wKey)=>{
    const b=[0,1].map(d=>p[bKey+d]);
    const w=[0,1].map(d=>x.map((_,j)=>p[wKey+d+j]));
    const terms=w.map(row2=>row2.map((wj,j)=>[wj,x[j],wj*x[j]]));
    const pre=terms.map(ts=>ts.reduce((s,t)=>s+t[2],0));
    return {b,w,terms,pre,value:pre.map((s,d)=>b[d]+s)};
  };
  const mu=head('mb','m'),vraw=head('vb','v');
  // Softplus acts on bias + weighted sum, exactly as in ToyVAE.encoder().
  const pre=vraw.value,act=pre.map(softplus),variance=act.map(v=>v+1e-4),sd=variance.map(Math.sqrt);
  const eps=model.noise(0)[i],z=mu.value.map((m,d)=>m+sd[d]*eps[d]);
  return {i,name:row.name,raw,x,obs:{s:[raw[0],raw[1]],u:[raw[2],raw[3]]},
    mu:{b:mu.b,w:mu.w,terms:mu.terms,sum:mu.pre,value:mu.value},
    variance:{b:vraw.b,w:vraw.w,terms:vraw.terms,sum:vraw.pre,pre,act,value:variance,sd},
    eps,z};
}
// Two principal directions of a small batch, by power iteration with deflation.
export function pca2(rows){
  const n=rows.length,d=rows[0].length;
  const mean=Array.from({length:d},(_,k)=>rows.reduce((s,r)=>s+r[k],0)/n);
  const centered=rows.map(r=>r.map((v,k)=>v-mean[k]));
  let M=Array.from({length:d},()=>Array(d).fill(0));
  for(const r of centered)for(let a=0;a<d;a++)for(let b=0;b<d;b++)M[a][b]+=r[a]*r[b]/n;
  const unit=v=>{const s=Math.hypot(...v);return s<1e-12?null:v.map(x=>x/s);};
  const matVec=(A,v)=>A.map(row=>row.reduce((s,a,j)=>s+a*v[j],0));
  const dot=(u,v)=>u.reduce((s,a,j)=>s+a*v[j],0);
  const axes=[],eigen=[];
  for(let k=0;k<2;k++){
    let v=unit(Array.from({length:d},(_,t)=>Math.sin((t+1)*(k+1)*1.7)+.01))??Array.from({length:d},(_,t)=>t===k?1:0);
    for(let it=0;it<300;it++){const nx=unit(matVec(M,v));if(!nx)break;v=nx;}
    const kk=v.reduce((best,t)=>Math.abs(v[t])>Math.abs(v[best])?t:best,0);
    if(v[kk]<0)v=v.map(x=>-x);
    axes.push(v);
    const lam=dot(v,matVec(M,v));
    eigen.push(lam);
    M=M.map((row,a)=>row.map((val,b)=>val-lam*v[a]*v[b]));
  }
  const total=eigen.reduce((s,v)=>s+v,0)||1;
  return {mean,axes,eigen,explained:eigen.map(v=>v/total),scores:centered.map(r=>axes.map(a=>dot(a,r)))};
}
// Pairwise distances in the 4-D encoder input versus the 2-D latent mean.
export function pairwiseGeometry(model){
  const trace=model.data.map((_,i)=>encoderTrace(model,i)),pairs=[];
  for(let i=0;i<trace.length;i++)for(let j=i+1;j<trace.length;j++){
    const dx=Math.hypot(...trace[i].x.map((v,k)=>v-trace[j].x[k]));
    const dm=Math.hypot(...trace[i].mu.value.map((v,k)=>v-trace[j].mu.value[k]));
    pairs.push({i,j,dx,dm});
  }
  const mx=pairs.reduce((s,p)=>s+p.dx,0)/pairs.length,my=pairs.reduce((s,p)=>s+p.dm,0)/pairs.length;
  let sxy=0,sxx=0,syy=0;
  for(const p of pairs){const a=p.dx-mx,b=p.dm-my;sxy+=a*b;sxx+=a*a;syy+=b*b;}
  return {pairs,r:sxx>0&&syy>0?sxy/Math.sqrt(sxx*syy):0,trace};
}
export function gradientCheck(){
  const model=new ToyVAE(42),r=model.evaluate({gradient:true});const checks=[];
  for(const key of ['m00','v00','t00','w00','beta0','gamma0']){
    const old=model.p[key],h=1e-5;model.p[key]=old+h;const plus=model.evaluate().loss;model.p[key]=old-h;const minus=model.evaluate().loss;model.p[key]=old;
    const numerical=(plus-minus)/(2*h),analytic=r.gradients[key],relativeError=Math.abs(numerical-analytic)/Math.max(1,Math.abs(numerical),Math.abs(analytic));checks.push({key,numerical,analytic,relativeError});
  }
  return checks;
}

// Nonlinear representation lab. This is a separate, synthetic VAE, not RegVelo:
// it reconstructs expression directly and deliberately omits the RNA ODE,
// normalisation layers, dropout and graph penalties. Shared scalar AD above.
export class NetworkLab {
  constructor(dim=2,logInput=false,{genes=12,cellCount=36,standardize=false,data=null}={}){
    this.dim=dim;this.hidden=8;this.genes=genes;this.input=genes*2;this.logInput=logInput;this.standardize=standardize;
    const rng=new RNG(811);
    this.data=data?.length?data.map((row,i)=>({name:row.name??row.id??`细胞 ${i+1}`,group:row.group??i%3,raw:row.raw.slice()})):Array.from({length:cellCount},(_,i)=>{
      const group=i%3,t=Math.floor(i/3)/Math.max(1,Math.ceil(cellCount/3)-1);
      const raw=Array.from({length:this.input},(_,j)=>{
        const gene=j%genes,branch=(group-1)*t;
        return .15+softplus(-.8+1.5*Math.sin(gene*.73+t*2.1)+branch*Math.cos(gene*1.3)+(j<genes?.35:-.2)+.05*rng.normal());
      });
      return {name:`细胞 ${i+1}`,group,raw};
    });
    this.kinetics=fitTwoPoolKinetics(this.data,{genes:this.genes});
    const transformed=this.data.map(c=>c.raw.map(v=>logInput?Math.log(.01+v):v));
    this.inputMean=Array.from({length:this.input},(_,j)=>standardize?transformed.reduce((s,r)=>s+r[j],0)/this.data.length:0);
    this.inputSD=this.inputMean.map((m,j)=>standardize?Math.max(1e-6,Math.sqrt(transformed.reduce((s,r)=>s+(r[j]-m)**2,0)/this.data.length)):1);
    this.reset();
  }
  reset(){
    const r=new RNG(927);this.p={};this.m={};this.v={};this.epoch=0;this.optimizerStep=0;this.parameterSteps={};this.trainingDraw=0;this.lastTrainingNoise=null;this.history=[];this.last=null;
    const layer=(tag,rows,cols,scale,bias)=>{for(let j=0;j<rows;j++){this.p[`${tag}b${j}`]=bias;for(let k=0;k<cols;k++)this.p[`${tag}${j}_${k}`]=r.normal()*scale;}};
    layer('h',8,this.input,.17,.25);layer('mu',this.dim,8,.2,0);layer('va',this.dim,8,.03,-2.5);
    layer('d',8,this.dim,.3,.35);layer('o',this.input,8,.2,.2);
    this.noise=this.data.map(()=>Array.from({length:this.dim},()=>r.normal()));
    const evaluationRng=new RNG(849671);
    this.evaluationNoise=Array.from({length:8},()=>this.data.map(()=>Array.from({length:this.dim},()=>evaluationRng.normal())));
  }
  sampleTrainingNoise(){const r=new RNG(604031+(++this.trainingDraw)*104729);this.lastTrainingNoise=this.data.map(()=>Array.from({length:this.dim},()=>r.normal()));return this.lastTrainingNoise;}
  assess({lambda=.05}={}){
    const rs=this.evaluationNoise.map(noise=>this.evaluate({lambda,noise}));
    const mean=key=>rs.reduce((sum,r)=>sum+(r[key]??0),0)/rs.length;
    return {loss:mean('loss'),reconstruction:mean('reconstruction'),kl:mean('kl'),regulation:mean('regulation'),gradients:null,samples:rs.length};
  }
  affine(tag,rows,x){return Array.from({length:rows},(_,j)=>x.reduce((s,v,k)=>s+v*this.p[`${tag}${j}_${k}`],this.p[`${tag}b${j}`]));}
  inputVector(raw){return raw.map((v,j)=>((this.logInput?Math.log(.01+v):v)-this.inputMean[j])/this.inputSD[j]);}
  decode(latent){
    const decoderPre=this.affine('d',8,latent),decoderHidden=decoderPre.map(v=>Math.max(0,v));
    const outputPre=this.affine('o',this.input,decoderHidden),prediction=outputPre.map(softplus);
    return {decoderPre,decoderHidden,outputPre,prediction};
  }
  trace(i,{gene=0,delta=0,serial=0,useMean=false,epsilon=null}={}){
    const raw=this.data[i].raw.slice();raw[gene]=Math.max(.001,raw[gene]+delta);
    const x=this.inputVector(raw),a=this.affine('h',8,x),h=a.map(v=>Math.max(0,v));
    const mu=this.affine('mu',this.dim,h),vraw=this.affine('va',this.dim,h),variance=vraw.map(v=>softplus(v)+1e-4);
    const r=new RNG(774+serial*131+i*83),eps=epsilon?epsilon.slice():serial?Array.from({length:this.dim},()=>r.normal()):this.noise[i].slice();
    const z=mu.map((v,j)=>v+Math.sqrt(variance[j])*eps[j]);
    const latent=(useMean?mu:z).slice(),decoded=this.decode(latent);
    return {raw,x,a,h,mu,vraw,variance,eps,z,latent,...decoded};
  }
  evaluate({lambda=.1,gradient=false,noise=this.noise}={}){
    if(!gradient){
      let reconstruction=0,kl=0;
      for(let i=0;i<this.data.length;i++){
        const r=this.trace(i,{useMean:false,epsilon:noise[i]});
        reconstruction+=r.prediction.reduce((s,v,j)=>s+(v-r.raw[j])**2/(2*.3**2)+Math.log(.3)+.5*Math.log(2*Math.PI),0)/this.data.length;
        kl+=.5*r.mu.reduce((s,v,j)=>s+v*v+r.variance[j]-Math.log(r.variance[j])-1,0)/this.data.length;
      }
      return {loss:reconstruction+lambda*kl,reconstruction,kl,gradients:null};
    }
    tape=[];
    const p=Object.fromEntries(Object.entries(this.p).map(([k,v])=>[k,av(v)]));
    const linear=(tag,n,x)=>Array.from({length:n},(_,j)=>x.reduce((s,v,k)=>add(s,mul(v,p[`${tag}${j}_${k}`])),p[`${tag}b${j}`]));
    const relu=a=>{const n=new A(Math.max(0,a.v),()=>{if(a.v>0)a.g+=n.g;});return n;};
    let rec=av(0),kl=av(0);
    this.data.forEach((row,i)=>{
      const x=this.inputVector(row.raw).map(av);
      const h=linear('h',8,x).map(relu),mu=linear('mu',this.dim,h),v=linear('va',this.dim,h).map(v=>add(sp(v),1e-4));
      const z=mu.map((m,j)=>add(m,mul(sqrt(v[j]),noise[i][j])));
      const pred=linear('o',this.input,linear('d',8,z).map(relu)).map(sp);
      rec=add(rec,sum(pred.map((v,j)=>add(div(sq(sub(v,row.raw[j])),2*.3**2),Math.log(.3)+.5*Math.log(2*Math.PI)))));
      kl=add(kl,mul(.5,sum(mu.map((m,j)=>sub(sub(add(sq(m),v[j]),log(v[j])),1)))));
    });
    rec=div(rec,this.data.length);kl=div(kl,this.data.length);const loss=add(rec,mul(lambda,kl));
    if(gradient)backward(loss);
    const result={loss:loss.v,reconstruction:rec.v,kl:kl.v,gradients:gradient?Object.fromEntries(Object.entries(p).map(([k,v])=>[k,v.g])):null};
    tape=[];return result;
  }
  parameterGroup(key){return /^(d|o)/.test(key)?'decoder':/^(mu|va)/.test(key)?'latent':'encoder';}
  step({lambda=.1,lr=.006,mode='joint'}={}){
    const noise=this.sampleTrainingNoise(),r=this.evaluate({lambda,gradient:true,noise});
    if(!Number.isFinite(r.loss)||Object.values(r.gradients).some(v=>!Number.isFinite(v)))throw Error('出现无效数值，请重置实验。');
    const enabled=key=>mode==='joint'||(mode==='encoder'&&this.parameterGroup(key)!=='decoder')||(mode==='encoderHidden'&&this.parameterGroup(key)==='encoder')||mode===this.parameterGroup(key);
    const norm=Math.hypot(...Object.entries(r.gradients).filter(([key])=>enabled(key)).map(([,v])=>v)),clip=Math.min(1,10/(norm+1e-12));this.optimizerStep++;this.epoch++;
    for(const k of Object.keys(this.p))if(enabled(k)){const g=r.gradients[k]*clip,n=(this.parameterSteps[k]??0)+1;this.parameterSteps[k]=n;this.m[k]=.9*(this.m[k]??0)+.1*g;this.v[k]=.999*(this.v[k]??0)+.001*g*g;this.p[k]-=lr*(this.m[k]/(1-.9**n))/(Math.sqrt(this.v[k]/(1-.999**n))+1e-8);}
    const after=this.evaluate({lambda,noise});this.last={...r,norm,mode,before:r.loss,after:after.loss,noiseSerial:this.trainingDraw};this.history.push({epoch:this.epoch,...after,trainingLoss:after.loss});return this.last;
  }
}

// Early stopping on this teaching experiment's deterministic training objective.
// The best checkpoint contains Adam moments/counters as well as weights, so
// continuing after restoration is consistent. epoch counts actual updates.
export class AutoFitSession {
  constructor(model,metrics,{minSteps=100,patience=100,maxSteps=1000,relativeTolerance=.001}={}){
    if(!Number.isFinite(metrics.loss))throw Error('当前目标无效，请先重置模型。');
    this.options={minSteps,patience,maxSteps,relativeTolerance};this.startEpoch=model.epoch;this.startLoss=metrics.loss;this.attempts=0;this.stale=0;this.anchor=metrics.loss;this.done=false;this.reason=null;this.record(model,metrics);
  }
  record(model,metrics){
    this.best={epoch:model.epoch,metrics:{loss:metrics.loss,reconstruction:metrics.reconstruction,kl:metrics.kl,regulation:metrics.regulation},p:{...model.p},m:{...model.m},v:{...model.v},optimizerStep:model.optimizerStep,parameterSteps:model.parameterSteps?{...model.parameterSteps}:null,trainingDraw:model.trainingDraw,lastTrainingNoise:model.lastTrainingNoise,last:model.last};
  }
  observe(model,metrics){
    if(this.done)return true;
    if(!Number.isFinite(metrics.loss))throw Error('目标出现无效数值，已停止自动优化。');
    this.attempts++;if(metrics.loss<this.best.metrics.loss)this.record(model,metrics);
    const {relativeTolerance,minSteps,patience,maxSteps}=this.options;
    if(this.anchor-metrics.loss>relativeTolerance*Math.max(1,Math.abs(this.anchor))){this.anchor=metrics.loss;this.stale=0;}else this.stale++;
    if(this.attempts>=maxSteps){this.done=true;this.reason='limit';}
    else if(this.attempts>=minSteps&&this.stale>=patience){this.done=true;this.reason='plateau';}
    return this.done;
  }
  finish(model,reason=this.reason??'paused'){
    const best=this.best,restored=Object.keys(best.p).some(k=>best.p[k]!==model.p[k]);
    model.p={...best.p};model.m={...best.m};model.v={...best.v};model.optimizerStep=best.optimizerStep;
    model.trainingDraw=best.trainingDraw;model.lastTrainingNoise=best.lastTrainingNoise;
    if(best.parameterSteps)model.parameterSteps={...best.parameterSteps};model.last=best.last;
    if(restored)model.history.push({epoch:model.epoch,...best.metrics,restored:true});
    this.done=true;this.reason=reason;
    return {reason,attempts:this.attempts,bestStep:best.epoch-this.startEpoch,bestEpoch:best.epoch,startLoss:this.startLoss,bestLoss:best.metrics.loss,restored,options:{...this.options}};
  }
}

// Joint dynamics teaching model. It shares the paper-toy observations with
// NetworkLab; the decoder predicts one time per gene and integrates the
// coupled ODE. No synthetic group labels or known times enter the loss.
const kineticOps={add:(a,b)=>a+b,sub:(a,b)=>a-b,mul:(a,b)=>a*b,div:(a,b)=>a/b,sp:softplus,av:x=>x};
const kineticADOps={add,sub,mul,div,sp,av};
function kineticFieldInternal(state,p,o,genes=3){
  const u=state.slice(0,genes),s=state.slice(genes),alpha=s.map((_,g)=>o.sp(s.reduce((v,sj,j)=>o.add(v,o.mul(p[`kw${g}_${j}`],sj)),p['kb'+g])));
  const beta=s.map((_,g)=>o.sp(p['kbeta'+g])),gamma=s.map((_,g)=>o.sp(p['kgamma'+g]));
  return [...u.map((v,g)=>o.sub(alpha[g],o.mul(beta[g],v))),...s.map((v,g)=>o.sub(o.mul(beta[g],u[g]),o.mul(gamma[g],v)))];
}
function kineticIntegrateInternal(time,p,steps,o,genes=3){
  let y=Array.from({length:2*genes},()=>o.av(0));const dt=o.div(time,steps);
  const shift=(state,k,f)=>state.map((v,j)=>o.add(v,o.mul(o.mul(dt,f),k[j])));
  for(let k=0;k<steps;k++){
    const a=kineticFieldInternal(y,p,o,genes),b=kineticFieldInternal(shift(y,a,.5),p,o,genes),c=kineticFieldInternal(shift(y,b,.5),p,o,genes),d=kineticFieldInternal(shift(y,c,1),p,o,genes);
    y=y.map((v,j)=>o.add(v,o.mul(o.div(dt,6),o.add(o.add(a[j],o.mul(2,b[j])),o.add(o.mul(2,c[j]),d[j])))));
  }
  return y;
}
export function kineticState(time,parameters,steps=12,genes=3){return kineticIntegrateInternal(time,parameters,steps,kineticOps,genes);}
export function kineticField(state,parameters,genes=3){return kineticFieldInternal(state,parameters,kineticOps,genes);}

export class DynamicsLab extends NetworkLab {
  constructor({data=null,genes=3}={}){super(3,false,{genes,cellCount:data?.length??24,standardize:true,data});this.genes=genes;this.tMax=4;this.odeSteps=12;this.sigma=.2;this.gradientBatchSize=Math.min(this.data.length,16);this.parameterSteps={};}
  reset(){
    super.reset();for(const key of Object.keys(this.p))if(/^o[0-9b]/.test(key))delete this.p[key];
    const rng=new RNG(20261026);
    for(let g=0;g<this.genes;g++){
      this.p['tmb'+g]=-.2+g*.12;
      for(let j=0;j<8;j++)this.p[`tm${g}_${j}`]=rng.normal()*.22;
      this.p['kb'+g]=invsp(.8+g*.15);this.p['kbeta'+g]=invsp(.8+g*.1);this.p['kgamma'+g]=invsp(.55+g*.08);
      for(let j=0;j<this.genes;j++)this.p[`kw${g}_${j}`]=rng.normal()*.09;
    }
    this.parameterSteps={};this.last=null;
  }
  rates(){return Array.from({length:this.genes},(_,g)=>({beta:softplus(this.p['kbeta'+g]),gamma:softplus(this.p['kgamma'+g]),bias:this.p['kb'+g],weights:Array.from({length:this.genes},(_,j)=>this.p[`kw${g}_${j}`])}));}
  decode(latent){
    const decoderPre=this.affine('d',8,latent),decoderHidden=decoderPre.map(v=>Math.max(0,v)),timePre=this.affine('tm',this.genes,decoderHidden),time=timePre.map(v=>this.tMax*sigmoid(v));
    const integratedStates=time.map(t=>kineticState(t,this.p,this.odeSteps,this.genes));
    const prediction=[...integratedStates.map((y,g)=>y[this.genes+g]),...integratedStates.map((y,g)=>y[g])];
    return {decoderPre,decoderHidden,timePre,time,integratedStates,prediction};
  }
  evaluate({lambda=.05,gradient=false,noise=this.noise}={}){
    if(!gradient){
      let reconstruction=0,kl=0;
      for(let i=0;i<this.data.length;i++){
        const r=this.trace(i,{useMean:false,epsilon:noise[i]});
        reconstruction+=r.prediction.reduce((sum,v,j)=>sum+(v-r.raw[j])**2/(2*this.sigma**2)+Math.log(this.sigma)+.5*Math.log(2*Math.PI),0)/this.data.length;
        kl+=.5*r.mu.reduce((sum,v,j)=>sum+v*v+r.variance[j]-Math.log(r.variance[j])-1,0)/this.data.length;
      }
      const regulation=.001*Object.entries(this.p).filter(([key])=>key.startsWith('kw')).reduce((sum,[,v])=>sum+v*v,0);
      return {loss:reconstruction+lambda*kl+regulation,reconstruction,kl,regulation,gradients:null};
    }
    tape=[];const p=Object.fromEntries(Object.entries(this.p).map(([key,v])=>[key,av(v)]));
    const linear=(tag,n,x)=>Array.from({length:n},(_,j)=>x.reduce((sum,v,k)=>add(sum,mul(v,p[`${tag}${j}_${k}`])),p[`${tag}b${j}`]));
    const relu=a=>{const n=new A(Math.max(0,a.v),()=>{if(a.v>0)a.g+=n.g;});return n;};
    let reconstruction=av(0),kl=av(0);
    const batchIndices=this.data.length>this.gradientBatchSize?Array.from({length:this.gradientBatchSize},(_,k)=>Math.floor(k*this.data.length/this.gradientBatchSize)):this.data.map((_,i)=>i);
    batchIndices.forEach(i=>{const row=this.data[i];
      const h=linear('h',8,this.inputVector(row.raw)).map(relu),mu=linear('mu',3,h),variance=linear('va',3,h).map(v=>add(sp(v),1e-4));
      const z=mu.map((v,j)=>add(v,mul(sqrt(variance[j]),noise[i][j]))),hidden=linear('d',8,z).map(relu),time=linear('tm',this.genes,hidden).map(v=>mul(this.tMax,sig(v)));
      const ys=time.map(t=>kineticIntegrateInternal(t,p,this.odeSteps,kineticADOps,this.genes)),prediction=[...ys.map((y,g)=>y[this.genes+g]),...ys.map((y,g)=>y[g])];
      reconstruction=add(reconstruction,sum(prediction.map((v,j)=>add(div(sq(sub(v,row.raw[j])),2*this.sigma**2),Math.log(this.sigma)+.5*Math.log(2*Math.PI)))));
      kl=add(kl,mul(.5,sum(mu.map((v,j)=>sub(sub(add(sq(v),variance[j]),log(variance[j])),1)))));
    });
    reconstruction=div(reconstruction,batchIndices.length);kl=div(kl,batchIndices.length);
    const regulation=mul(.001,sum(Object.keys(p).filter(key=>key.startsWith('kw')).map(key=>sq(p[key])))),loss=add(add(reconstruction,mul(lambda,kl)),regulation);
    backward(loss);
    const result={loss:loss.v,reconstruction:reconstruction.v,kl:kl.v,regulation:regulation.v,gradients:Object.fromEntries(Object.entries(p).map(([key,v])=>[key,v.g]))};tape=[];return result;
  }
  parameterGroup(key){return key.startsWith('kbeta')?'beta':key.startsWith('kgamma')?'gamma':key.startsWith('k')?'grn':/^(d|tm)/.test(key)?'time':'encoder';}
  step({lambda=.05,lr=.008,mode='joint'}={}){
    const noise=this.sampleTrainingNoise(),result=this.evaluate({lambda,gradient:true,noise}),before={...this.p};
    const enabled=key=>mode==='joint'||(mode==='rates'&&['beta','gamma','grn'].includes(this.parameterGroup(key)))||mode===this.parameterGroup(key);
    if(!Number.isFinite(result.loss)||Object.values(result.gradients).some(v=>!Number.isFinite(v)))throw Error('动力学出现非有限数值，请重置本页模型。');
    const norm=Math.hypot(...Object.entries(result.gradients).filter(([key])=>enabled(key)).map(([,v])=>v)),clip=Math.min(1,10/(norm+1e-12));
    for(const key of Object.keys(this.p))if(enabled(key)){
      const g=result.gradients[key]*clip,n=(this.parameterSteps[key]??0)+1;this.parameterSteps[key]=n;
      this.m[key]=.9*(this.m[key]??0)+.1*g;this.v[key]=.999*(this.v[key]??0)+.001*g*g;
      this.p[key]-=lr*(this.m[key]/(1-.9**n))/(Math.sqrt(this.v[key]/(1-.999**n))+1e-8);
    }
    this.epoch++;const after=this.evaluate({lambda,noise}),groups={};
    for(const group of ['encoder','time','grn','beta','gamma']){
      const keys=Object.keys(this.p).filter(key=>this.parameterGroup(key)===group);
      groups[group]={gradientNorm:Math.hypot(...keys.map(key=>result.gradients[key])),updateNorm:Math.hypot(...keys.map(key=>this.p[key]-before[key])),enabled:keys.some(enabled)};
    }
    this.last={...result,before:result.loss,after:after.loss,groups,mode,noiseSerial:this.trainingDraw,changes:Object.fromEntries(Object.keys(this.p).map(key=>[key,{before:before[key],after:this.p[key],gradient:result.gradients[key]}]))};
    this.history.push({epoch:this.epoch,...after,trainingLoss:after.loss});return this.last;
  }
}
