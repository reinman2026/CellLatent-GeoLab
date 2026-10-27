const dot3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const sub3=(a,b)=>a.map((v,i)=>v-b[i]);
const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];

function inverse(matrix){
  const n=matrix.length,a=matrix.map((row,i)=>[...row,...Array.from({length:n},(_,j)=>+(i===j))]);
  for(let j=0;j<n;j++){
    let pivot=j;for(let i=j+1;i<n;i++)if(Math.abs(a[i][j])>Math.abs(a[pivot][j]))pivot=i;
    if(Math.abs(a[pivot][j])<1e-14){for(let i=0;i<n;i++)a[i][i]+=1e-8; pivot=j;for(let i=j+1;i<n;i++)if(Math.abs(a[i][j])>Math.abs(a[pivot][j]))pivot=i;}
    if(Math.abs(a[pivot][j])<1e-14)throw Error('潜空间几何矩阵不可逆。');
    [a[j],a[pivot]]=[a[pivot],a[j]];const scale=a[j][j];for(let k=0;k<2*n;k++)a[j][k]/=scale;
    for(let i=0;i<n;i++)if(i!==j){const q=a[i][j];for(let k=0;k<2*n;k++)a[i][k]-=q*a[j][k];}
  }
  return a.map(row=>row.slice(n));
}

function eigenSymmetric3(matrix){
  const a=matrix.map(row=>row.slice()),vectors=[[1,0,0],[0,1,0],[0,0,1]];
  for(let it=0;it<48;it++){
    let p=0,q=1;if(Math.abs(a[0][2])>Math.abs(a[p][q]))[p,q]=[0,2];if(Math.abs(a[1][2])>Math.abs(a[p][q]))[p,q]=[1,2];
    if(Math.abs(a[p][q])<1e-13)break;
    const angle=.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(angle),s=Math.sin(angle);
    const app=c*c*a[p][p]-2*s*c*a[p][q]+s*s*a[q][q],aqq=s*s*a[p][p]+2*s*c*a[p][q]+c*c*a[q][q];
    for(let k=0;k<3;k++)if(k!==p&&k!==q){const kp=a[k][p],kq=a[k][q];a[k][p]=a[p][k]=c*kp-s*kq;a[k][q]=a[q][k]=s*kp+c*kq;}
    a[p][p]=app;a[q][q]=aqq;a[p][q]=a[q][p]=0;
    for(let k=0;k<3;k++){const kp=vectors[k][p],kq=vectors[k][q];vectors[k][p]=c*kp-s*kq;vectors[k][q]=s*kp+c*kq;}
  }
  const order=[0,1,2].sort((i,j)=>a[i][i]-a[j][j]);
  return {values:order.map(i=>a[i][i]),vectors:vectors.map(row=>order.map(i=>row[i]))};
}

// Incrementally extend an outward-oriented tetrahedron across visible faces.
export function convexHull3D(points){
  const n=points.length;if(n<4)return {faces:[],edges:[]};
  const ranges=[0,1,2].map(j=>Math.max(...points.map(p=>p[j]))-Math.min(...points.map(p=>p[j])));
  const tolerance=Math.max(1,Math.hypot(...ranges))*1e-9;
  const i0=points.reduce((a,p,i)=>p[0]<points[a][0]?i:a,0),i1=points.reduce((a,p,i)=>p[0]>points[a][0]?i:a,0),line=sub3(points[i1],points[i0]),line2=dot3(line,line);
  const i2=points.reduce((best,p,i)=>{if(i===i0||i===i1)return best;const delta=sub3(p,points[i0]),cross=cross3(line,delta),distance=dot3(cross,cross)/line2;return distance>best.distance?{index:i,distance}:best;},{index:-1,distance:-1}).index;
  if(i2<0)return {faces:[],edges:[]};
  const normal=cross3(sub3(points[i1],points[i0]),sub3(points[i2],points[i0])),normalLength=Math.sqrt(dot3(normal,normal));
  const i3=points.reduce((best,p,i)=>{if([i0,i1,i2].includes(i))return best;const distance=Math.abs(dot3(normal,sub3(p,points[i0])))/normalLength;return distance>best.distance?{index:i,distance}:best;},{index:-1,distance:-1}).index;
  if(i3<0)return {faces:[],edges:[]};
  const interior=[i0,i1,i2,i3].reduce((acc,i)=>acc.map((v,j)=>v+points[i][j]/4),[0,0,0]);
  const makeFace=(a,b,c)=>{let ids=[a,b,c],norm=cross3(sub3(points[b],points[a]),sub3(points[c],points[a]));if(dot3(norm,sub3(interior,points[a]))>0){ids=[a,c,b];norm=norm.map(v=>-v);}return {ids,norm,offset:-dot3(norm,points[ids[0]])};};
  let faces=[[i0,i1,i2],[i0,i3,i1],[i1,i3,i2],[i2,i3,i0]].map(f=>makeFace(...f));
  const initial=new Set([i0,i1,i2,i3]);
  for(let index=0;index<n;index++){
    if(initial.has(index))continue;
    const visible=faces.filter(face=>(dot3(face.norm,points[index])+face.offset)/Math.max(1e-14,Math.sqrt(dot3(face.norm,face.norm)))>tolerance);
    if(!visible.length)continue;
    const edges=new Map();
    for(const face of visible)for(let e=0;e<3;e++){const a=face.ids[e],b=face.ids[(e+1)%3],key=a<b?a+':'+b:b+':'+a,entry=edges.get(key);if(entry)entry.count++;else edges.set(key,{a,b,count:1});}
    const visibleSet=new Set(visible);faces=faces.filter(face=>!visibleSet.has(face));
    for(const edge of edges.values())if(edge.count===1)faces.push(makeFace(edge.a,edge.b,index));
  }
  const edgeSet=new Set();for(const face of faces)for(let e=0;e<3;e++){const a=face.ids[e],b=face.ids[(e+1)%3];edgeSet.add(a<b?a+':'+b:b+':'+a);}
  return {faces:faces.map(face=>face.ids),edges:[...edgeSet].map(edge=>edge.split(':').map(Number))};
}

// Khachiyan's iterative algorithm computes a minimum-volume ellipsoid that
// encloses all points. The output is geometric and has no probability meaning.
export function minimumVolumeEnclosingEllipsoid3D(points,{tolerance=1e-2,maxIterations=5000}={}){
  const n=points.length,d=3;if(n<d+1)throw Error('至少需要四个三维点。');
  const q=points.map(p=>[...p,1]),weights=Array(n).fill(1/n);let converged=false,iterations=0;
  for(;iterations<maxIterations;iterations++){
    const moment=Array.from({length:4},()=>Array(4).fill(0));
    for(let i=0;i<n;i++)for(let a=0;a<4;a++)for(let b=0;b<4;b++)moment[a][b]+=weights[i]*q[i][a]*q[i][b];
    let inv;try{inv=inverse(moment);}catch{break;}const scores=q.map(row=>row.reduce((sum,x,a)=>sum+x*row.reduce((v,y,b)=>v+inv[a][b]*y,0),0));
    let index=0;for(let i=1;i<n;i++)if(scores[i]>scores[index])index=i;
    const max=scores[index];if(max<=d+1+tolerance){converged=true;break;}
    const step=(max-d-1)/((d+1)*(max-1));for(let i=0;i<n;i++)weights[i]*=1-step;weights[index]+=step;
  }
  const center=points[0].map((_,j)=>weights.reduce((sum,w,i)=>sum+w*points[i][j],0));
  const covariance=Array.from({length:3},(_,a)=>Array.from({length:3},(_,b)=>weights.reduce((sum,w,i)=>sum+w*(points[i][a]-center[a])*(points[i][b]-center[b]),0)));for(let i=0;i<3;i++)covariance[i][i]+=1e-4;
  let matrix=inverse(covariance).map(row=>row.map(v=>v/d)),eigen=eigenSymmetric3(matrix),axes=eigen.values.map(v=>1/Math.sqrt(Math.max(v,1e-15)));
  let scores=points.map(point=>{const x=sub3(point,center);return x.reduce((sum,a,i)=>sum+a*x.reduce((v,b,j)=>v+matrix[i][j]*b,0),0);});
  const enclosureScale=Math.max(1,Math.sqrt(Math.max(...scores)));matrix=matrix.map(row=>row.map(v=>v/(enclosureScale*enclosureScale)));axes=axes.map(v=>v*enclosureScale);scores=scores.map(v=>v/(enclosureScale*enclosureScale));
  const extent=[0,1,2].map(j=>Math.sqrt(eigen.values.reduce((sum,value,k)=>sum+eigen.vectors[j][k]**2/Math.max(value,1e-15),0)));
  return {center,matrix,axes,vectors:eigen.vectors,extent,converged,iterations,maxScore:Math.max(...scores),scores};
}
