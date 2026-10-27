import {NetworkLab,RNG,twoPoolVelocity,AutoFitSession} from './core.mjs';
import {verifiedSource} from './verified-source.mjs';
import {paperToyData} from './paper-toy-data.mjs';

// Surface of (z-mu)^T diag(v)^-1 (z-mu) = k^2; radii use SD, not variance.
export function ellipsoidPoint(mu,variance,k,latitude,longitude){
  const unit=[Math.cos(latitude)*Math.cos(longitude),Math.cos(latitude)*Math.sin(longitude),Math.sin(latitude)];
  return mu.map((m,j)=>m+k*Math.sqrt(variance[j])*unit[j]);
}
export function rotateLatent(v,yaw,pitch){
  const c=Math.cos(yaw),s=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x=c*v[0]+s*v[2],z=-s*v[0]+c*v[2];
  return [x,cp*v[1]-sp*z,sp*v[1]+cp*z];
}
export function projectEllipsoid(mu,variance,k,yaw,pitch){
  const basis=[0,1,2].map(j=>rotateLatent([+(j===0),+(j===1),+(j===2)],yaw,pitch));
  const a=k*k*variance.reduce((s,v,j)=>s+v*basis[j][0]**2,0);
  const b=-k*k*variance.reduce((s,v,j)=>s+v*basis[j][0]*basis[j][1],0);
  const c=k*k*variance.reduce((s,v,j)=>s+v*basis[j][1]**2,0);
  const gap=Math.hypot(a-c,2*b),center=rotateLatent(mu,yaw,pitch);
  return {center,covariance:[[a,b],[b,c]],radii:[Math.sqrt(Math.max(0,(a+c+gap)/2)),Math.sqrt(Math.max(0,(a+c-gap)/2))],angle:.5*Math.atan2(2*b,a-c),extent:[Math.sqrt(a),Math.sqrt(c)]};
}

// Rotate the projected frame so the longest axis of the reference ellipsoid
// is horizontal and its second-longest axis is vertical on the canvas.
export function orientLatentProjection(mu,variance,yaw,pitch){
  const shape=projectEllipsoid(mu,variance,1,yaw,pitch),c=Math.cos(-shape.angle),s=Math.sin(-shape.angle);
  return {shape,rotate:v=>[c*v[0]-s*v[1],s*v[0]+c*v[1]]};
}

// Small deterministic UMAP-style projection for the teaching page. It builds
// a fuzzy k-nearest-neighbor graph in 3D, then relaxes a 2D spring layout.
export function umapEmbedding(points,{neighbors=5,iterations=120}={}){
  const n=points.length;if(!n)return [];
  const center=Array.from({length:3},(_,j)=>points.reduce((s,p)=>s+p[j],0)/n);
  const centered=points.map(p=>p.map((v,j)=>v-center[j]));
  const dot=(a,b)=>a.reduce((s,v,j)=>s+v*b[j],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=v=>Math.hypot(...v)||1;
  const axis=seed=>{let v=Array.from({length:3},(_,j)=>Math.sin((j+1)*(seed+1)*1.37)+.2);for(let t=0;t<80;t++){const next=Array(3).fill(0);for(let i=0;i<3;i++)for(let j=0;j<3;j++)next[i]+=centered.reduce((s,p)=>s+p[i]*p[j],0)*v[j];const size=Math.hypot(...next)||1;v=next.map(x=>x/size);}return v;};
  const e0=axis(0),e0n=e0.map(v=>v/norm(e0));let seed=[0,1,0];if(Math.abs(dot(seed,e0n))>.9)seed=[1,0,0];const e1o=seed.map((v,j)=>v-e0n[j]*dot(seed,e0n)),e1n=e1o.map(v=>v/norm(e1o)),e2=cross(e0n,e1n);
  let xy=centered.map(p=>[dot(p,e0n),dot(p,e1n)]);const distances=Array.from({length:n},()=>Array(n).fill(0));
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){const d=Math.hypot(...centered[i].map((v,k)=>v-centered[j][k]));distances[i][j]=distances[j][i]=d;}
  const graph=Array.from({length:n},()=>Array(n).fill(0));
  for(let i=0;i<n;i++){
    const order=Array.from({length:n},(_,j)=>j).filter(j=>j!==i).sort((a,b)=>distances[i][a]-distances[i][b]).slice(0,Math.min(neighbors,n-1));
    const rho=distances[i][order[0]]??0,sigma=Math.max(1e-5,order.reduce((s,j)=>s+distances[i][j],0)/Math.max(1,order.length));
    for(const j of order)graph[i][j]=Math.exp(-Math.max(0,distances[i][j]-rho)/sigma);
  }
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)graph[i][j]=graph[j][i]=Math.max(graph[i][j],graph[j][i]);
  const span=Math.max(1,...xy.flat().map(Math.abs));xy=xy.map(p=>p.map(v=>v/span));
  for(let it=0;it<iterations;it++){
    const next=xy.map(p=>p.slice()),rate=.12*(1-it/iterations);
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){
      const dx=xy[j][0]-xy[i][0],dy=xy[j][1]-xy[i][1],d=Math.hypot(dx,dy)||1e-4;
      const target=Math.min(1.8,Math.max(.08,distances[i][j]/(span||1))),w=graph[i][j],spring=w*(d-target)*.035;
      const rep=(1-w)*.003/(d*d+.08),force=(spring-rep)/d;
      next[i][0]+=dx*force*rate;next[i][1]+=dy*force*rate;next[j][0]-=dx*force*rate;next[j][1]-=dy*force*rate;
    }
    xy=next;
  }
  const max=Math.max(1,...xy.flat().map(Math.abs));return xy.map(p=>p.map(v=>v/max));
}

export function createSpatialLab(){
  const $=id=>document.getElementById(id),f=(v,n=3)=>Number(v).toFixed(n),vec=v=>'['+v.map(x=>f(x)).join(', ')+']';
  const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const genes=paperToyData.genes.length,inputDim=paperToyData.featureOrder.length,labels=paperToyData.featureOrder,colors=['#07888b','#775acb','#cf7852'];
  const createModel=()=>new NetworkLab(3,false,{genes,cellCount:paperToyData.cells.length,standardize:true,data:paperToyData.cells});
  let model=createModel(),selected=0,view='overview',representation='mean',ellipsoidScope='all',contourScale=1,yaw=-.6,pitch=.25;
  let sampleSerial=0,showCloud=false,step=0,unit=0,feature=0,delta=0,ghost=null,busy=false,timer=null,playTimer=null,lambda=.05,drawHits=[],umapHits=[],lastUmap=[],lossNow=null,status='';
  let pointer=null,dragged=false,lastRows=[],autoSession=null,autoResult=null,trainingEpochs=1000,trainingMode='joint';
  const tuneNames={joint:'联合拟合',encoder:'编码器',latent:'潜分布 μ / v',decoder:'解码器'};
  const names={overview:['同一个细胞，穿过三个空间','把一行十二维表达编码成三维潜变量，再由解码器还原为十二维表达。三个视图中的编号始终对应同一个细胞。'],encoder:['逐层看见，十二个数怎样变成三个数','跟着选中的细胞，查看输入缩放、隐藏层、三维分布参数与解码结果。所有算术使用同一份当前模型。'],training:['拟合改变网络，也改变细胞的位置','先预测十二维表达，再与观测比较，沿计算图反传梯度。联合拟合会同时改变潜分布和重建结果，也可单独微调模块。'],space:['每个细胞，都是一个三维椭球','均值决定椭球的中心，三个标准差决定三条半轴的长度。旋转视图，观察不同细胞的分布位置、形状与重叠。'],construction:['这份十二维实验，怎样对应 RegVelo','统一用十二维输入和三维潜变量解释结构；再与真实源码中的时间解码和 RNA 动力学逐段对照。']};
  const stageNames=['读取十二维表达','缩放到网络输入','编码隐藏层','三维潜变量','解码隐藏层','重建十二维表达'];
  function traces(){return model.data.map((_,i)=>model.trace(i,{useMean:representation==='mean',serial:sampleSerial,...(i===selected?{gene:feature,delta}:{})}));}
  function source(key){const s=verifiedSource.excerpts[key];return `<details class="sp-source"><summary>源码：${s.path.split('/').at(-1)}:${s.start}</summary><pre><code>${esc(s.text)}</code></pre></details>`;}
  function trainingControls(){return `<div class="fit-actions" role="group" aria-label="模型更新"><label for="sp-mode">范围<select id="sp-mode">${Object.entries(tuneNames).map(([m,n])=>`<option value="${m}">${n}</option>`).join('')}</select></label><label for="sp-epochs">epoch 轮数<select id="sp-epochs">${[100,500,1000,3000,10000].map(n=>`<option value="${n}">${n}</option>`).join('')}</select></label><button id="sp-train-one">更新 1 轮</button><button id="sp-train">按轮数训练</button><button id="sp-auto-fit" class="primary" title="以所选轮数为上限，按独立多抽样评估保留最佳模型">自动优化</button><button id="sp-stop">暂停</button><button id="sp-reset">重置模型</button></div>`;}
  function tuneButton(mode,label){return `<button data-sp-tune="${mode}" class="module-tune" title="对全部细胞微调${label}，其余参数固定">微调${label} · 10 轮</button>`;}
  function template(route='overview'){
    view=route;const [title,desc]=names[route];
    return `<section class="sp-hero"><div><span class="eyebrow">REGVELO · 统一的三空间实验室</span><h1>${title}</h1><p>${desc}</p></div><div class="sp-dimensions"><div><b>${inputDim}</b><span>原始表达维度</span></div><i>→</i><div class="latent"><b>3</b><span>潜变量维度</span></div><i>→</i><div><b>${inputDim}</b><span>重建表达维度</span></div></div></section>
    <div class="sp-controls fit-toolbar"><div class="sp-select-cell"><span>当前细胞</span><button id="sp-prev" aria-label="上一个细胞">←</button><select id="sp-cell" aria-label="选择细胞">${model.data.map((r,i)=>`<option value="${i}">C${String(i+1).padStart(2,'0')} · 教学组 ${r.group+1}</option>`).join('')}</select><button id="sp-next" aria-label="下一个细胞">→</button></div><div class="sp-view-options"><label>送入解码器<select id="sp-repr"><option value="mean">三维均值 μ</option><option value="sample">三维样本 z</option></select></label><button id="sp-resample">重新抽样 ε</button><span id="sp-epoch" class="tag"></span><span id="sp-best-step" class="tag fit-best" hidden></span>${trainingControls()}</div><div id="sp-fit-status" class="fit-status" role="status" aria-live="polite"></div></div>
    <section class="sp-spaces sp-stacked" aria-label="同一细胞的输入、潜变量和重建结果">
      <article class="sp-space-card sp-input"><header><div><span class="sp-kicker">01 / EXPRESSION</span><h2>原始表达空间 <small>ℝ¹²</small></h2></div><span class="sp-corner">观测 x</span>${tuneButton("encoder","编码器")}</header><p class="sp-card-note">${genes} 个基因 × 已剪接 / 未剪接 RNA，共 ${inputDim} 维</p><canvas id="sp-input-chart" class="sp-chart" role="img" aria-label="所有细胞的十二维原始表达平行坐标图"></canvas><div class="sp-vector" id="sp-input-vector"></div><footer>${inputDim} 条轴表示完整的 ${inputDim} 维状态；一条折线是一个细胞。</footer></article>
      <div class="sp-bridge"><span>编码器</span><b>→</b><small>12 → 8 → 3</small></div>
      <article class="sp-space-card sp-latent"><header><div><span class="sp-kicker">02 / LATENT</span><h2>三维潜空间 <small>ℝ³</small></h2></div><span class="sp-corner">编码结果</span>${tuneButton("latent","μ / v")}</header><div class="sp-sphere-controls"><div class="sp-segments"><button id="sp-all-ellipsoids">全部椭球</button><button id="sp-selected-ellipsoid">仅选中</button></div><select id="sp-contour" aria-label="椭球轮廓大小"><option value="1">1σ 轮廓</option><option value="2">2σ 轮廓</option><option value="2.795483482915">95% 概率</option></select><button id="sp-focus" aria-label="恢复默认观察角度">复位视角</button></div><canvas id="sp-latent-chart" class="sp-chart sp-sphere-chart" role="img" tabindex="0" aria-label="每个细胞的三维高斯椭球；中心由均值决定，半轴由标准差决定；拖动旋转，点击选择细胞"></canvas><div class="sp-vector" id="sp-latent-vector"></div><footer id="sp-ellipsoid-note"></footer></article>
      <div class="sp-bridge"><span>解码器</span><b>→</b><small>3 → 8 → 12</small></div>
      <article class="sp-space-card sp-output"><header><div><span class="sp-kicker">03 / RECONSTRUCTION</span><h2>重建数据空间 <small>ℝ¹²</small></h2></div><span class="sp-corner">预测 x̂</span>${tuneButton("decoder","解码器")}</header><p class="sp-card-note">与左侧完全相同的十二个特征和刻度</p><canvas id="sp-output-chart" class="sp-chart" role="img" aria-label="解码器输出的十二维表达，虚线对照同一细胞的原始表达"></canvas><div class="sp-vector" id="sp-output-vector"></div><footer>实线是重建，虚线是当前细胞的观测；两线越近，误差越小。</footer></article>
    </section>
    <div class="sp-legend"><span><i style="background:#07888b"></i>教学组 1</span><span><i style="background:#775acb"></i>教学组 2</span><span><i style="background:#cf7852"></i>教学组 3</span><span><i class="selected"></i>选中细胞</span><span>椭球中心 = 均值 · 半轴 = k × 标准差 · 拖动旋转 / 点击选择</span></div>
    <div class="sp-insight" id="sp-correspondence" role="status"></div>
    <div class="sp-section-tabs" aria-label="讲解页面"><a href="#overview">三空间总览</a><a href="#encoder">逐层计算</a><a href="#training">拟合过程</a><a href="#space">潜空间与分布</a><a href="#kinetics">两池与潜时间</a><a href="#construction">RegVelo 对应</a></div>
    ${route==='encoder'?encodingSection()+comparisonSection()+trainingSection():route==='training'?trainingSection()+comparisonSection()+encodingSection():route==='space'?distributionSection()+comparisonSection()+encodingSection():route==='construction'?referenceSection()+simulationFrameworkSection()+comparisonSection():comparisonSection()+encodingSection()+trainingSection()}
    <section class="sp-footnote"><b>实验设定</b><p>全站共用 ${paperToyData.cells.length} 个 paper toy 细胞、${genes} 个基因、${inputDim} 维表达与 3 维潜变量。颜色用于跟踪教学分组，不参与训练。本实验直接重建表达；RegVelo 的实际生成过程包含时间解码与 RNA 动力学，见“RegVelo 对应”。</p></section>`;
  }
  function simulationFrameworkSection(){return `<section class="sp-panel sp-framework" id="sp-sim-framework"><div class="sp-panel-title"><div><span class="sp-kicker">FINAL FRAMEWORK · SIMULATION</span><h2>模拟实验的完整闭环</h2><p>这张图把模拟模型的前向、损失、反向传播和参数更新放在同一条线上。它没有 W，也没有 ODE；最后一步直接把三维潜变量解码成 ${inputDim} 维表达。</p></div><span class="fit-location-note">与上方训练按钮共用当前模型</span></div><div class="sp-framework-flow"><div class="sp-framework-node"><b>输入 x</b><small>${inputDim} 维 S/U</small></div><i>→</i><div class="sp-framework-node"><b>Encoder</b><small>${inputDim} → 8 → μ,v</small></div><i>→</i><div class="sp-framework-node"><b>重参数化</b><small>z = μ + √v ⊙ ε</small></div><i>→</i><div class="sp-framework-node"><b>Decoder</b><small>3 → 8 → ${inputDim}</small></div><i>→</i><div class="sp-framework-node"><b>重建 x̂</b><small>${inputDim} 维表达</small></div></div><div class="sp-framework-loss"><span>x 与 x̂</span><b>→</b><span>ℒ = 重建项 + 0.05 × KL</span><b>→</b><span>反向传播</span><b>→</b><span>更新 h / mu / va / d / o</span><b>↺</b><span>下一轮</span></div><div class="sp-framework-groups"><div><b>编码器参数</b><span>h：8×${inputDim} + 8 个偏置</span><span>mu：3×8 + 3 个偏置</span><span>va：3×8 + 3 个偏置</span></div><div><b>解码器参数</b><span>d：8×3 + 8 个偏置</span><span>o：${inputDim}×8 + ${inputDim} 个偏置</span></div><div><b>固定输入</b><span>${paperToyData.cells.length} 个 toy 细胞</span><span>${inputDim} 维观测顺序保持不变</span></div></div><p class="sp-framework-note">每个 epoch（训练轮次）先对全部细胞做一次前向，再把总目标沿计算图反传；联合模式同时更新五组参数。切换顶部范围可以只更新编码器、潜分布或解码器。</p></section>`;}
  function comparisonSection(){return `<section class="sp-panel"><div class="sp-panel-title"><div><span class="sp-kicker">READ THE SAME CELL</span><h2>把同一个细胞的十二个表达值放在一起</h2><p>每一行都对照同一特征。表达误差来自解码器的实际输出。</p></div><div class="sp-error-badge"><span>当前细胞均方误差</span><b id="sp-mse"></b></div></div><div class="sp-comparison" id="sp-comparison"></div><div class="sp-perturb"><label>尝试改变一个输入 <select id="sp-feature">${labels.map((x,i)=>`<option value="${i}">${x.replace('_',' · ')}</option>`).join('')}</select></label><label class="sp-slider"><span>表达增量 <output id="sp-delta-output"></output></span><input id="sp-delta" type="range" min="-0.6" max="0.6" step="0.02" value="0"></label><button id="sp-restore">恢复观测</button></div><p class="sub" id="sp-perturb-note">输入改动只用于前向演示，不改写训练数据。</p></section>`;}
  function encodingSection(){return `<section class="sp-panel" id="sp-encoding-panel"><div class="sp-panel-title"><div><span class="sp-kicker">FOLLOW ONE FORWARD PASS</span><h2>从十二个输入，到三个潜坐标，再回到十二个输出</h2><p>Encoder（编码器）压缩细胞状态；Decoder（解码器）从潜坐标生成表达预测。</p></div><div class="button-row">${tuneButton("encoder","编码器")}<button id="sp-step">下一步</button><button id="sp-play">自动走一遍</button></div></div><div class="sp-stages">${stageNames.map((s,i)=>`<button data-sp-step="${i}"><b>0${i+1}</b><span>${s}</span></button>`).join('')}</div><div class="sp-stage-explanation" id="sp-stage-explanation" aria-live="polite"></div><div class="sp-arithmetic"><div><h3>选一个编码隐藏单元</h3><div class="sp-unit-buttons">${Array.from({length:8},(_,i)=>`<button data-sp-unit="${i}">h${i+1}</button>`).join('')}</div><div id="sp-unit-equation" class="sp-equation"></div><div class="table-scroll"><table id="sp-products"></table></div></div><div><h3>三个潜坐标与解码输入</h3><div id="sp-latent-equation" class="sp-equation"></div><p class="sub">ReLU（整流线性函数）把负输入置零；Softplus（平滑正值函数）保证方差和重建均值为正。缩放统计来自整批观测，在训练时固定。</p><div id="sp-decoder-equation" class="sp-equation"></div></div></div></section>`;}
  function trainingSection(){return `<section class="sp-panel" id="sp-training-panel"><div class="sp-panel-title"><div><span class="sp-kicker">LEARN THE MAPPING</span><h2>更新一次网络，三个空间会发生什么？</h2><p>原始观测固定，编码器和解码器共同更新。三幅图都读取同一份当前参数。</p></div><span class="fit-location-note">使用顶部工具栏更新模型</span></div><div class="sp-training-metrics"><div><span>参数更新</span><b id="sp-updates"></b></div><div><span>总目标</span><b id="sp-loss"></b></div><div><span>重建项</span><b id="sp-rec"></b></div><div><span>KL 分布约束</span><b id="sp-kl"></b></div><div><span>全体细胞重建 MSE</span><b id="sp-all-mse"></b></div></div><canvas id="sp-loss-chart" class="sp-loss-chart" role="img" aria-label="真实训练的总目标与重建误差曲线"></canvas><div class="sp-legend"><span><i style="background:#07888b"></i>总目标</span><span><i style="background:#cf7852"></i>重建项</span><span>每点采用固定 8 组评估抽样，训练使用固定噪声序列</span></div><p id="sp-training-status" class="status" role="status"></p><div class="sp-equation">ℒ = 平均负对数似然 + 0.05 × KL(q(z | x) ‖ 𝒩(0,I))</div><p class="sub">epoch（训练轮次）表示全部 300 个细胞完成一轮更新。每轮使用固定的三维 ε 序列；损失比较全部十二个表达分量。预览中的“重抽样”只切换显示点。固定 8 组独立抽样用于比较模型，仍评估同一批细胞，不是独立验证集。</p><p class="sub">KL divergence（KL 散度）约束每个细胞的分布接近标准正态先验。训练通过重参数化对隐藏层和解码器求导，使用 Adam（自适应优化器）更新权重。连续概率密度可大于 1，负对数似然因此可为负。</p></section>`;}
  function distributionSection(){return `<section class="sp-panel"><div class="sp-panel-title"><div><span class="sp-kicker">POSITION AND UNCERTAINTY</span><h2>均值决定位置，方差决定三维椭球的形状</h2><p>每个半透明椭球对应一个细胞的近似后验。中心点始终位于 μ；切换解码样本只会改变粉色样本点。</p></div><label class="check"><input type="checkbox" id="sp-cloud">显示当前细胞的 60 个后验样本</label></div><div class="sp-distribution-grid"><div><h3>当前细胞的三维近似后验</h3><div class="sp-equation" id="sp-posterior-equation"></div><p>Posterior（后验）描述给定表达后，潜变量可能的取值。对角高斯的椭球沿三条潜坐标轴伸展，半轴使用标准差 √v。</p></div><div><h3>等密度椭球的方程</h3><div class="sp-equation" id="sp-ellipsoid-equation"></div><p>三个方差相同时，轮廓是球形；方差不同时，轮廓沿不确定性更大的方向拉长。拖动旋转改变观察角度，均值与方差保持不变。</p></div></div><div class="sp-kinetics-wrap"><div class="sp-umap-heading"><div><h3>先拟合两池动力学参数</h3><p>按 RegVelo 的 two-pool（两池）方程，从时间排序的 S/U 表达学习每个基因的 α、β、γ。</p></div><span class="sp-umap-note">α 转录 · β 剪接 · γ 降解</span></div><table id="sp-kinetics-table" class="sp-kinetics"><thead><tr><th>基因</th><th>α₀ + 调控项</th><th>β</th><th>γ</th><th>样本对</th></tr></thead><tbody></tbody></table><div class="sp-equation">du<sub>g</sub>/dt = α<sub>g</sub>(s) − β<sub>g</sub>u<sub>g</sub>，&nbsp; ds<sub>g</sub>/dt = β<sub>g</sub>u<sub>g</sub> − γ<sub>g</sub>s<sub>g</sub></div></div><div class="sp-umap-wrap"><div class="sp-umap-heading"><div><h3>UMAP 二维轨迹与速度方向</h3><p>UMAP（统一流形近似与投影）把三维均值的邻域关系压到二维。线段按教学时间顺序连接；箭头使用两池动力学计算出的表达速度，再经过编码器映射到潜空间。</p></div><span class="sp-umap-note">动力学参数驱动的潜空间速度</span></div><canvas id="sp-umap-chart" class="sp-umap-chart" role="img" aria-label="三维潜均值的二维 UMAP 投影、细胞轨迹和两池动力学速度方向"></canvas><div class="sp-umap-legend"><span><i style="background:#07888b"></i>教学组 1</span><span><i style="background:#775acb"></i>教学组 2</span><span><i style="background:#cf7852"></i>教学组 3</span><span>箭头 = 两池方程计算的潜空间速度方向</span></div></div><div class="sp-note">三维高斯的 1σ 椭球内概率约为 19.9%，2σ 约为 73.9%。选择“95% 概率”时，三条半轴统一乘以 2.795。轮廓以外仍然存在可能的样本，椭球不代表分布的硬边界。</div></section>`;}
  function referenceSection(){return `<section class="sp-panel"><div class="sp-panel-title"><div><span class="sp-kicker">LINK TO REGVELO</span><h2>统一尺寸解释机制，保留真实模型的区别</h2><p>页面所有可操作示例固定为 12 → 3 → 12。下表解释本实验与已核对源码的对应关系。</p></div></div><div class="table-scroll"><table class="sp-reference"><thead><tr><th>步骤</th><th>本页真实计算</th><th>RegVelo 源码</th></tr></thead><tbody><tr><td>表达输入</td><td>6 个基因的 S/U，共 12 维；固定标准化</td><td>拼接 S/U 为 2G 维；可选 log 变换</td></tr><tr><td>编码器</td><td>12 → 8 个 ReLU 单元 → μ、v 各 3 维</td><td>默认 256 个隐藏单元，μ、v 默认各 10 维；含归一化与随机失活</td></tr><tr><td>潜变量</td><td>三维对角高斯；可看 μ 或抽样 z</td><td>默认正态配置同样给出均值、方差与样本</td></tr><tr><td>生成路径</td><td>3 → 8 → 12，直接生成表达均值</td><td>z → 逐基因时间 → 调控网络与 RNA 动力学 → 表达分布</td></tr><tr><td>拟合目标</td><td>十二维高斯重建项 + 加权 KL</td><td>还包含由配置决定的调控、速度与偏置约束</td></tr></tbody></table></div><div class="sp-equation">RegVelo：du<sub>g</sub>/dt = α<sub>g</sub>(s) − β<sub>g</sub>u<sub>g</sub>，ds<sub>g</sub>/dt = β<sub>g</sub>u<sub>g</sub> − γ<sub>g</sub>s<sub>g</sub></div><p>ODE（常微分方程）描述转录、剪接与降解怎样连续改变表达。本页简化解码器以便逐项展示算术；这些教学权重和细胞不代表真实 RegVelo 的生物学拟合结果。</p>${source('input')}${source('heads')}${source('time')}${source('loss')}</section>`;}

  function canvas(id){const c=$(id),w=c.clientWidth,h=c.clientHeight,d=window.devicePixelRatio||1;c.width=Math.round(w*d);c.height=Math.round(h*d);const g=c.getContext('2d');g.scale(d,d);g.font='11px "Microsoft YaHei",sans-serif';return {c,g,w,h};}
  function drawParallel(id,rows,isOutput,scales){
     const {c,g,w,h}=canvas(id),left=30,right=20,top=30,bottom=43,X=j=>left+j*(w-left-right)/Math.max(1,labels.length-1),Y=(v,j)=>h-bottom-(h-top-bottom)*v/scales[j];
    const hits=[];g.textAlign='center';
    for(let j=0;j<labels.length;j++){g.strokeStyle='#dbe4e8';g.lineWidth=1;g.beginPath();g.moveTo(X(j),top);g.lineTo(X(j),h-bottom);g.stroke();g.fillStyle='#637481';g.fillText(labels[j].replace('_',''),X(j),h-19);g.font='9px "Microsoft YaHei"';g.fillText(f(scales[j],1),X(j),top-10);g.fillText('0',X(j),h-bottom+12);g.font='11px "Microsoft YaHei"';}
    const stroke=(values,color,width,alpha,dash=[])=>{g.save();g.globalAlpha=alpha;g.strokeStyle=color;g.lineWidth=width;g.setLineDash(dash);g.beginPath();values.forEach((v,j)=>j?g.lineTo(X(j),Y(v,j)):g.moveTo(X(j),Y(v,j)));g.stroke();g.restore();};
    rows.forEach((r,i)=>{const values=isOutput?r.prediction:r.raw;if(i!==selected)stroke(values,colors[model.data[i].group],1,.17);values.forEach((v,j)=>hits.push({i,x:X(j),y:Y(v,j)}));for(let j=0;j<labels.length-1;j++)hits.push({i,x:(X(j)+X(j+1))/2,y:(Y(values[j],j)+Y(values[j+1],j+1))/2});});
    const r=rows[selected],values=isOutput?r.prediction:r.raw;
    if(isOutput)stroke(r.raw,'#72818d',1.7,.8,[4,4]);else if(delta)stroke(model.data[selected].raw,'#72818d',1.5,.7,[4,4]);
    stroke(values,colors[model.data[selected].group],2.7,1);
    values.forEach((v,j)=>{g.beginPath();g.arc(X(j),Y(v,j),4,0,Math.PI*2);g.fillStyle=colors[model.data[selected].group];g.fill();g.strokeStyle='white';g.lineWidth=1.5;g.stroke();});
    c._spHits=hits;
  }
  function rotate(v){return rotateLatent(v,yaw,pitch);}
  function drawLatent(rows){
    const {g,w,h}=canvas('sp-latent-chart');
    const indices=ellipsoidScope==='all'?rows.map((_,i)=>i):[selected];
    const shapes=indices.map(i=>({i,...projectEllipsoid(rows[i].mu,rows[i].variance,contourScale,yaw,pitch)}));
    const samples=[];
    if(showCloud&&view==='space'){
      const rng=new RNG(880+sampleSerial*47),row=rows[selected];
      for(let k=0;k<60;k++)samples.push(row.mu.map((m,j)=>m+Math.sqrt(row.variance[j])*rng.normal()));
    }
    const extent=Math.max(.1,...indices.flatMap(i=>rows[i].mu.map((m,j)=>Math.abs(m)+contourScale*Math.sqrt(rows[i].variance[j]))));
    const axisEnds=[0,1,2].map(j=>{const p=[0,0,0];p[j]=extent;return p;});
    let xmin=0,xmax=0,ymin=0,ymax=0;
    const include=(v,dx=0,dy=0)=>{xmin=Math.min(xmin,v[0]-dx);xmax=Math.max(xmax,v[0]+dx);ymin=Math.min(ymin,v[1]-dy);ymax=Math.max(ymax,v[1]+dy);};
    shapes.forEach(p=>include(p.center,...p.extent));axisEnds.forEach(v=>include(rotate(v)));
    samples.forEach(v=>include(rotate(v)));
    if(representation==='sample')include(rotate(rows[selected].z));
    if(ghost){const r=ghost[selected],shape=projectEllipsoid(r.mu,r.variance,contourScale,yaw,pitch);include(shape.center,...shape.extent);}
    const scale=Math.min((w-58)/Math.max(.1,xmax-xmin),(h-64)/Math.max(.1,ymax-ymin));
    const cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;
    const screenRot=v=>({x:w/2+(v[0]-cx)*scale,y:h/2-(v[1]-cy)*scale,depth:v[2]});
    const project=v=>screenRot(rotate(v));
    const line=(pts,color,width=1,alpha=1,dash=[])=>{g.save();g.globalAlpha=alpha;g.strokeStyle=color;g.lineWidth=width;g.setLineDash(dash);g.beginPath();pts.forEach((v,j)=>{const p=project(v);j?g.lineTo(p.x,p.y):g.moveTo(p.x,p.y);});g.stroke();g.restore();};
    for(const t of [0,.5,1]){
      const v=t*extent;
      line([[0,0,v],[extent,0,v]],'#dfe6ee',.7);
      line([[v,0,0],[v,0,extent]],'#dfe6ee',.7);
    }
    axisEnds.forEach((v,j)=>{line([[0,0,0],v],['#69a3a1','#9a87c3','#bd9273'][j],1);const p=project(v);g.fillStyle='#718293';g.textAlign='center';g.fillText('z'+(j+1),p.x,p.y-5);});
    const drawShape=(shape,old=false)=>{
      const {i}=shape,active=i===selected&&!old,r=rows[i],p=screenRot(shape.center),color=old?'#8c99a7':active?'#bd8725':colors[model.data[i].group];
      g.save();g.translate(p.x,p.y);g.rotate(shape.angle);g.beginPath();g.ellipse(0,0,shape.radii[0]*scale,shape.radii[1]*scale,0,0,Math.PI*2);
      if(!old){g.fillStyle=color;g.globalAlpha=active?.12:.025;g.fill();}
      g.globalAlpha=old?.6:active?.95:.27;g.strokeStyle=color;g.lineWidth=active?1.6:.8;g.setLineDash(old?[3,3]:[]);g.stroke();g.restore();
      if(old)return;
      const param=(lat,lon)=>ellipsoidPoint(r.mu,r.variance,contourScale,lat,lon);
      // Wire curves lie on the same 3-D ellipsoid as the projected silhouette.
      for(const lat of active?[0]:[0])line(Array.from({length:49},(_,k)=>param(lat,k*Math.PI/24)),color,.65,active?.42:.13);
      for(const lon of active?[0,Math.PI/2]:[0,Math.PI/2])line(Array.from({length:33},(_,k)=>param(-Math.PI/2+k*Math.PI/16,lon)),color,.65,active?.42:.13);
    };
    shapes.sort((a,b)=>a.center[2]-b.center[2]);
    shapes.filter(p=>p.i!==selected).forEach(p=>drawShape(p));
    if(ghost){const r=ghost[selected];drawShape({i:selected,...projectEllipsoid(r.mu,r.variance,contourScale,yaw,pitch)},true);}
    drawShape(shapes.find(p=>p.i===selected));
    samples.forEach(v=>{const p=project(v);g.fillStyle='#b26799';g.globalAlpha=.5;g.beginPath();g.arc(p.x,p.y,1.8,0,Math.PI*2);g.fill();});g.globalAlpha=1;
    drawHits=shapes.map(shape=>{
      const p=screenRot(shape.center),active=shape.i===selected;
      g.fillStyle=active?'#bd8725':colors[model.data[shape.i].group];g.globalAlpha=active?1:.75;g.beginPath();g.arc(p.x,p.y,active?4.5:2.5,0,Math.PI*2);g.fill();g.globalAlpha=1;
      if(active){g.strokeStyle='white';g.lineWidth=1.5;g.stroke();g.fillStyle='#8b651c';g.textAlign='left';g.font='600 11px "Microsoft YaHei"';g.fillText('C'+String(shape.i+1).padStart(2,'0')+' · μ',p.x+8,p.y-7);}
      return {i:shape.i,...p,rx:shape.radii[0]*scale,ry:shape.radii[1]*scale,angle:shape.angle};
    });
    if(representation==='sample'){
      const r=rows[selected],p=project(r.z);line([r.mu,r.z],'#b26799',1,.8,[3,3]);g.fillStyle='#ac4d83';g.beginPath();g.arc(p.x,p.y,4,0,Math.PI*2);g.fill();g.textAlign='left';g.fillText('z',p.x+7,p.y+3);
    }
    g.textAlign='center';g.fillStyle='#7e8b9b';g.font='10px "Microsoft YaHei"';g.fillText('中心 μ · 三条半轴 '+f(contourScale,contourScale===1||contourScale===2?0:3)+'√v · 三轴等比例',w/2,h-9);
  }
  function drawUmap(rows){
    if(!$('sp-umap-chart'))return;
    const {c,g,w,h}=canvas('sp-umap-chart'),points=rows.map(r=>r.mu),xy=umapEmbedding(points,{neighbors:6,iterations:points.length>120?20:120}),pad=34;
    lastUmap=xy.map((p,i)=>({i,xy:p}));const max=Math.max(.8,...xy.flat().map(v=>Math.abs(v))),X=v=>pad+(v+max)/(2*max)*(w-pad*2),Y=v=>h-pad-(v+max)/(2*max)*(h-pad*2);
    g.strokeStyle='#e3e8ed';g.lineWidth=1;g.beginPath();g.moveTo(pad,h/2);g.lineTo(w-pad,h/2);g.moveTo(w/2,pad);g.lineTo(w/2,h-pad);g.stroke();
    g.fillStyle='#8998a2';g.font='10px "Microsoft YaHei"';g.textAlign='left';g.fillText('UMAP₁',w-pad-38,h-10);g.fillText('UMAP₂',pad+4,pad-9);
    const projectVelocity=(i,v)=>{
      const nearest=Array.from({length:xy.length},(_,j)=>j).filter(j=>j!==i).sort((a,b)=>Math.hypot(...points[a].map((x,k)=>x-points[i][k]))-Math.hypot(...points[b].map((x,k)=>x-points[i][k]))).slice(0,4);
      let vx=0,vy=0,total=0;for(const j of nearest){const d=points[j].map((x,k)=>x-points[i][k]),den=d.reduce((s,x)=>s+x*x,0);if(den<1e-8)continue;const coeff=Math.max(-1.5,Math.min(1.5,v.reduce((s,x,k)=>s+x*d[k],0)/den)),weight=1/(Math.hypot(...d)+.05);vx+=(xy[j][0]-xy[i][0])*coeff*weight;vy+=(xy[j][1]-xy[i][1])*coeff*weight;total+=weight;}return total?[vx/total,vy/total]:[0,0];
    };
    const velocities=rows.map((r,i)=>{
      const expressionVelocity=twoPoolVelocity(r.raw,model.kinetics,{genes}),base=r.mu,eps=.001,latentVelocity=Array(3).fill(0);
      for(let j=0;j<model.input;j++){const d=expressionVelocity[j]*eps,shift=model.trace(i,{gene:j,delta:d,useMean:true}).mu;for(let k=0;k<3;k++)latentVelocity[k]+=(shift[k]-base[k])/eps;}
      return projectVelocity(i,latentVelocity);
    });lastUmap.forEach((p,i)=>p.velocity=velocities[i]);
    for(const group of [0,1,2]){const path=rows.map((r,i)=>({r,i})).filter(x=>x.r.group===group).sort((a,b)=>a.i-b.i);g.strokeStyle=colors[group];g.globalAlpha=.48;g.lineWidth=1.6;g.beginPath();path.forEach((x,k)=>{const px=X(xy[x.i][0]),py=Y(xy[x.i][1]);k?g.lineTo(px,py):g.moveTo(px,py);});g.stroke();g.globalAlpha=1;}
    lastUmap.forEach((p,i)=>{const x=X(p.xy[0]),y=Y(p.xy[1]),v=p.velocity,vm=Math.hypot(...v),color=colors[rows[i].group];if(vm>1e-6){const len=Math.min(28,Math.max(8,vm*75)),ux=v[0]/vm,uy=-v[1]/vm,ex=x+ux*len,ey=y+uy*len;g.strokeStyle=color;g.globalAlpha=.75;g.lineWidth=i===selected?2:1.2;g.beginPath();g.moveTo(x,y);g.lineTo(ex,ey);g.stroke();g.beginPath();g.moveTo(ex,ey);g.lineTo(ex-ux*5+uy*3,ey-uy*5-ux*3);g.lineTo(ex-ux*5-uy*3,ey-uy*5+ux*3);g.closePath();g.fillStyle=color;g.fill();g.globalAlpha=1;}g.fillStyle=color;g.beginPath();g.arc(x,y,i===selected?5:3.2,0,Math.PI*2);g.fill();g.strokeStyle='white';g.lineWidth=1;g.stroke();if(i===selected){g.fillStyle='#76551b';g.font='600 10px "Microsoft YaHei"';g.textAlign='left';g.fillText('C'+String(i+1).padStart(2,'0'),x+7,y-7);}});
    umapHits=lastUmap.map(p=>({i:p.i,x:X(p.xy[0]),y:Y(p.xy[1]),r:p.i===selected?9:7}));
  }
  function drawLoss(){
    if(!$('sp-loss-chart'))return;const {g,w,h}=canvas('sp-loss-chart'),hist=model.history;
    if(!hist.length){g.fillStyle='#788b95';g.fillText('点击“按轮数训练”，记录每次更新后的实际目标。',20,35);return;}
    const vals=hist.flatMap(r=>[r.loss,r.reconstruction]),min=Math.min(...vals),max=Math.max(...vals),lo=min-.08*(max-min+1),hi=max+.08*(max-min+1),X=n=>50+(w-70)*(n-hist[0].epoch)/Math.max(1,hist.at(-1).epoch-hist[0].epoch),Y=v=>h-32-(h-55)*(v-lo)/(hi-lo);
    for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;g.strokeStyle='#e7ecee';g.beginPath();g.moveTo(50,Y(v));g.lineTo(w-20,Y(v));g.stroke();g.fillStyle='#7a8c98';g.textAlign='right';g.fillText(f(v,1),43,Y(v)+4);}
    for(const [key,color] of [['loss','#07888b'],['reconstruction','#cf7852']]){g.strokeStyle=color;g.lineWidth=2;g.beginPath();hist.forEach((r,i)=>i?g.lineTo(X(r.epoch),Y(r[key])):g.moveTo(X(r.epoch),Y(r[key])));g.stroke();if(hist.length===1){g.fillStyle=color;g.beginPath();g.arc(X(hist[0].epoch),Y(hist[0][key]),3,0,7);g.fill();}}
    g.fillStyle='#7a8c98';g.textAlign='left';g.fillText(String(hist[0].epoch),50,h-12);g.textAlign='right';g.fillText('更新 '+hist.at(-1).epoch+' 次',w-20,h-12);
  }
  const stageDetails=[
    r=>`读取 C${String(selected+1).padStart(2,'0')} 的完整表达：x = ${vec(r.raw)}。前六项是 6 个基因的已剪接 RNA，后六项是未剪接 RNA。`,
    r=>`逐特征使用整批数据的固定均值与标准差缩放：x̃ = (x − m) / s = ${vec(r.x)}。这一步仍是十二维，尚未压缩。`,
    r=>`隐藏层先计算 Wₑ x̃ + bₑ，再取 ReLU：h = ${vec(r.h)}。每个隐藏单元都组合六项输入。`,
    r=>`μ = ${vec(r.mu)}，v = ${vec(r.variance)}。当前送给解码器的是${representation==='mean'?'均值 μ':'样本 z = μ + √v ⊙ ε'}，具体向量为 ${vec(r.latent)}。`,
    r=>`解码器读取上一步完整的三维向量，经过 8 个 ReLU 单元：hᴰ = ${vec(r.decoderHidden)}。椭球用于展示分布，解码器读取均值或本次抽样的三维坐标。`,
    r=>`输出层生成六项重建均值：x̂ = ${vec(r.prediction)}。它们与输入使用同一特征顺序，可逐项计算误差。`
  ];
  function refreshDetail(r){
    if($('sp-comparison')){
      const scale=Math.max(...r.raw,...r.prediction,.1)*1.06;
      $('sp-comparison').innerHTML=`<div class="sp-compare-heading"><span>表达特征</span><span>观测 / 重建</span><span>数值对照</span><span>差值 x̂ − x</span></div>`+labels.map((label,j)=>`<div class="sp-compare-row"><b>${label.replace('_',' · ')}</b><div class="sp-paired-bars"><i style="width:${r.raw[j]/scale*100}%;background:${colors[model.data[selected].group]}"></i><i style="width:${r.prediction[j]/scale*100}%;background:#97a2d0"></i></div><span>${f(r.raw[j])} <small>→</small> ${f(r.prediction[j])}</span><span class="${r.prediction[j]>r.raw[j]?'positive':'negative'}">${r.prediction[j]>=r.raw[j]?'+':''}${f(r.prediction[j]-r.raw[j])}</span></div>`).join('');
      $('sp-mse').textContent=f(r.raw.reduce((s,v,j)=>s+(v-r.prediction[j])**2,0)/model.input,4);$('sp-feature').value=feature;$('sp-delta').value=delta;$('sp-delta-output').textContent=(delta>=0?'+':'')+f(delta,2);
      $('sp-perturb-note').textContent=delta?`当前预览：只将 ${labels[feature]} 改动 ${f(delta,2)}（表达最低取 0.001）。两侧折线与潜向量均已重算；训练数据保持原值。`:'输入改动只用于前向演示，不改写训练数据。';
    }
    if($('sp-stage-explanation')){
      $('sp-stage-explanation').textContent=stageDetails[step](r);document.querySelectorAll('[data-sp-step]').forEach(b=>{b.classList.toggle('active',+b.dataset.spStep===step);b.setAttribute('aria-pressed',String(+b.dataset.spStep===step));});
      document.querySelectorAll('[data-sp-unit]').forEach(b=>b.classList.toggle('active',+b.dataset.spUnit===unit));
      const products=r.x.map((v,j)=>v*model.p[`h${unit}_${j}`]),bias=model.p['hb'+unit];
      $('sp-unit-equation').textContent=`h${unit+1} = ReLU(Σ W${unit+1},j × x̃j + b) = ReLU(${f(products.reduce((a,b)=>a+b,0))} + ${f(bias)}) = ${f(r.h[unit])}`;
      $('sp-products').innerHTML=`<thead><tr><th>输入</th><th>缩放后 x̃</th><th>权重 W</th><th>乘积</th></tr></thead><tbody>${labels.map((x,j)=>`<tr><td>${x}</td><td>${f(r.x[j])}</td><td>${f(model.p[`h${unit}_${j}`])}</td><td>${f(products[j])}</td></tr>`).join('')}</tbody>`;
      $('sp-latent-equation').innerHTML=`μ = ${vec(r.mu)}<br>v = ${vec(r.variance)}<br>ε = ${vec(r.eps)}<br>z = μ + √v ⊙ ε = ${vec(r.z)}<br><b>实际解码输入 = ${vec(r.latent)}</b>`;
      $('sp-decoder-equation').innerHTML=`hᴰ = ReLU(Wᴰ ${representation==='mean'?'μ':'z'} + bᴰ)<br>x̂ = Softplus(Wₒ hᴰ + bₒ)<br>x̂ = ${vec(r.prediction)}`;
    }
    if($('sp-posterior-equation')){
      $('sp-cloud').checked=showCloud;$('sp-posterior-equation').innerHTML=`q(z | x) = 𝒩(μ, diag(v))<br>μ = ${vec(r.mu)}<br>v = ${vec(r.variance)}<br>标准差 √v = ${vec(r.variance.map(Math.sqrt))}`;
      $('sp-ellipsoid-equation').innerHTML=`Σⱼ (zⱼ − μⱼ)² / vⱼ = k²<br>k = ${f(contourScale)}<br>中心 = ${vec(r.mu)}<br>三条半轴 = k√v = ${vec(r.variance.map(v=>contourScale*Math.sqrt(v)))}`;
    }
    if($('sp-kinetics-table'))$('sp-kinetics-table').querySelectorAll('tbody')[0].innerHTML=model.kinetics.map((p,j)=>`<tr><td>基因 ${paperToyData.genes[j]}</td><td>${f(p.alpha)}</td><td>${f(p.beta)}</td><td>${f(p.gamma)}</td><td>${p.fit.samples}</td></tr>`).join('');
  }
  function refresh(recompute=false){
    if(recompute||!lossNow)lossNow=model.assess({lambda});lastRows=traces();const r=lastRows[selected],c=model.data[selected],id='C'+String(selected+1).padStart(2,'0');
    $('sp-cell').value=selected;$('sp-repr').value=representation;$('sp-resample').disabled=representation==='mean';$('sp-epoch').textContent='已更新 '+model.epoch+' 次';
    $('sp-best-step').hidden=!autoSession&&!autoResult;$('sp-best-step').textContent=autoSession?`本轮最佳 第 ${autoSession.best.epoch-autoSession.startEpoch} 步`:autoResult?`本轮最佳 第 ${autoResult.bestStep} 步`:'';
    $('sp-mode').value=trainingMode;$('sp-epochs').value=trainingEpochs;
    ['sp-train','sp-train-one','sp-auto-fit','sp-mode','sp-epochs'].forEach(id=>$(id).disabled=busy);$('sp-stop').disabled=!busy;
    document.querySelectorAll('[data-sp-tune]').forEach(b=>b.disabled=busy);
    $('sp-fit-status').textContent=autoSession?`自动优化中 · 已计算 ${autoSession.attempts}/${autoSession.options.maxSteps} 轮 · 本轮最佳第 ${autoSession.best.epoch-autoSession.startEpoch} 轮。可随时暂停并保留最佳参数。`:autoResult?autoSummary():status||`训练噪声序列固定；可训练 ${trainingEpochs} 轮，或在该上限内自动选择最佳模型。各模块提供 10 轮微调。`;
    for(const [name,values,ls] of [['input',r.raw,labels],['latent',[...r.mu,...r.variance.map(Math.sqrt)],['μ₁','μ₂','μ₃','σ₁','σ₂','σ₃']],['output',r.prediction,labels]])$('sp-'+name+'-vector').innerHTML=`<b style="color:${colors[c.group]}">${id}</b>`+values.map((v,j)=>`<span><small>${ls[j]}</small>${f(v)}</span>`).join('');
    $('sp-all-ellipsoids').classList.toggle('active',ellipsoidScope==='all');$('sp-selected-ellipsoid').classList.toggle('active',ellipsoidScope==='selected');
    $('sp-all-ellipsoids').setAttribute('aria-pressed',String(ellipsoidScope==='all'));$('sp-selected-ellipsoid').setAttribute('aria-pressed',String(ellipsoidScope==='selected'));$('sp-contour').value=String(contourScale);
    const probability=contourScale===1?'19.9%':contourScale===2?'73.9%':'95.0%';
    $('sp-ellipsoid-note').textContent=`中心是均值 μ；标准差 σ = √v，半轴为 ${f(contourScale,contourScale===1||contourScale===2?0:3)} × σ。三维分布在此轮廓内的概率约 ${probability}。${representation==='sample'?'粉色点为本次样本 z。':''}`;
    $('sp-correspondence').innerHTML=`<strong>${id}</strong><span>十二维观测</span><b>→</b><span>${representation==='mean'?'μ':'z'} = ${vec(r.latent)}</span><b>→</b><span>十二维重建</span><small>点击图中的任一细胞，可跟踪同一条计算路径。</small>`;
    // Keep both expression plots on fixed, shared axes as training changes.
    const scales=labels.map((_,j)=>Math.max(.1,...model.data.map(r=>r.raw[j]))*1.5);
    drawParallel('sp-input-chart',lastRows,false,scales);drawLatent(lastRows);drawUmap(lastRows);drawParallel('sp-output-chart',lastRows,true,scales);refreshDetail(r);
    if($('sp-loss')){
      $('sp-updates').textContent=model.epoch;$('sp-loss').textContent=f(lossNow.loss);$('sp-rec').textContent=f(lossNow.reconstruction);$('sp-kl').textContent=f(lossNow.kl);
      $('sp-all-mse').textContent=f(lastRows.reduce((sum,r)=>sum+r.raw.reduce((s,v,j)=>s+(v-r.prediction[j])**2,0)/model.input,0)/lastRows.length,4);
      $('sp-training-status').textContent=status||'开始时为随机参数；点击训练，观察真实目标与细胞表示的变化。';
      drawLoss();
    }
  }
  function choose(i){selected=(i+model.data.length)%model.data.length;delta=0;refresh();}
  function stopPlayback(){if(playTimer!==null)clearTimeout(playTimer);playTimer=null;if($('sp-play'))$('sp-play').textContent='自动走一遍';}
  function autoSummary(){const r=autoResult,reason=r.reason==='plateau'?'改善趋缓，已停止':r.reason==='limit'?'已达到计算上限':r.reason==='error'?'出现异常，已停止':'已暂停';return `${reason} · 本轮最佳：第 ${r.bestStep} 步（累计第 ${r.bestEpoch} 次）· 实际计算 ${r.attempts} 步 · L ${f(r.startLoss,4)} → ${f(r.bestLoss,4)}。已保留最佳参数；仅依据当前教学数据的训练目标。`;}
  function finishAuto(reason){if(!autoSession)return;autoResult=autoSession.finish(model,reason);autoSession=null;lossNow=model.assess({lambda});status=autoSummary();}
  function stop(){if(autoSession)finishAuto('paused');else if(busy)status=`已暂停于第 ${model.epoch} 次更新，模型与细胞选择已保留。`;if(timer!==null)clearTimeout(timer);timer=null;busy=false;stopPlayback();}
  function train(count,{automatic=false}={}){
    stop();autoResult=null;delta=0;ghost=traces().map(r=>({mu:r.mu.slice(),z:r.z.slice(),variance:r.variance.slice()}));busy=true;const baseline=model.assess({lambda}),before=baseline.loss;let left=count;
    autoSession=automatic?new AutoFitSession(model,baseline,{maxSteps:count}):null;
    const run=()=>{try{const steps=Math.min(4,left);for(let k=0;k<steps;k++){
      model.step({lambda,lr:.008,mode:trainingMode});const now=model.assess({lambda}),record=model.history.at(-1);record.loss=now.loss;record.reconstruction=now.reconstruction;record.kl=now.kl;left--;
      if(autoSession?.observe(model,now)){finishAuto(autoSession.reason);break;}
    }busy=left>0&&!autoResult;timer=null;
    if(!autoResult)status=`${tuneNames[trainingMode]} · 本轮 ${count-left}/${count} 轮 · 8 组抽样评估 ${f(before)} → ${f(model.history.at(-1).loss)}。训练噪声序列固定；${busy?'可以暂停。':'更新完成。'} 潜空间灰色虚线椭球显示本轮开始前的分布。`;
    refresh(true);if(busy)timer=setTimeout(run,25);}catch(e){if(autoSession)finishAuto('error');stop();status=e.message+(autoResult?' '+autoSummary():'');refresh(true);}};refresh();timer=setTimeout(run,0);
  }
  function setup(route){
    view=route;$('sp-cell').onchange=()=>choose(+$('sp-cell').value);$('sp-prev').onclick=()=>choose(selected-1);$('sp-next').onclick=()=>choose(selected+1);
    $('sp-repr').onchange=()=>{representation=$('sp-repr').value;ghost=null;refresh();};$('sp-resample').onclick=()=>{sampleSerial++;ghost=null;refresh();};
    $('sp-all-ellipsoids').onclick=()=>{ellipsoidScope='all';refresh();};$('sp-selected-ellipsoid').onclick=()=>{ellipsoidScope='selected';refresh();};
    $('sp-contour').onchange=()=>{contourScale=Number($('sp-contour').value);refresh();};
    $('sp-focus').onclick=()=>{yaw=-.6;pitch=.25;drawLatent(lastRows);};
    const pick=(hits,event)=>{const b=event.currentTarget.getBoundingClientRect(),x=event.clientX-b.left,y=event.clientY-b.top;const nearest=hits.map(p=>({...p,d:(p.x-x)**2+(p.y-y)**2})).sort((a,b)=>a.d-b.d)[0];if(nearest&&nearest.d<240){choose(nearest.i);return;}const inside=hits.filter(p=>{if(!p.rx||!p.ry)return false;const dx=x-p.x,dy=y-p.y,c=Math.cos(p.angle),s=Math.sin(p.angle);return ((c*dx+s*dy)/p.rx)**2+((-s*dx+c*dy)/p.ry)**2<=1;}).sort((a,b)=>b.depth-a.depth);if(inside.length)choose(inside[0].i);};
    for(const id of ['sp-input-chart','sp-output-chart'])$(id).onclick=e=>pick(e.currentTarget._spHits,e);
    const c=$('sp-latent-chart');c.onpointerdown=e=>{pointer={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY};dragged=false;c.setPointerCapture?.(e.pointerId);};
    c.onpointermove=e=>{if(!pointer)return;const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;if(Math.hypot(e.clientX-pointer.startX,e.clientY-pointer.startY)>3)dragged=true;if(dragged){yaw+=dx*.004;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.004));drawLatent(lastRows);}pointer.x=e.clientX;pointer.y=e.clientY;};
    c.onpointerup=e=>{if(pointer&&!dragged)pick(drawHits,e);pointer=null;};c.onpointercancel=()=>{pointer=null;};
    c.onkeydown=e=>{const d={ArrowLeft:[-.06,0],ArrowRight:[.06,0],ArrowUp:[0,-.06],ArrowDown:[0,.06]}[e.key];if(d){e.preventDefault();yaw+=d[0];pitch=Math.max(-1.5,Math.min(1.5,pitch+d[1]));drawLatent(lastRows);}};
    if($('sp-umap-chart'))$('sp-umap-chart').onclick=e=>{const b=e.currentTarget.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top,hit=umapHits.map(p=>({...p,d:(p.x-x)**2+(p.y-y)**2})).sort((a,b)=>a.d-b.d)[0];if(hit&&hit.d<180)choose(hit.i);};
    if($('sp-feature')){$('sp-feature').onchange=()=>{feature=+$('sp-feature').value;delta=0;refresh();};$('sp-delta').oninput=()=>{delta=+$('sp-delta').value;refresh();};$('sp-restore').onclick=()=>{delta=0;refresh();};}
    if($('sp-step')){
      $('sp-step').onclick=()=>{stopPlayback();step=(step+1)%6;refresh();};$('sp-play').onclick=()=>{if(playTimer!==null){stopPlayback();return;}step=0;refresh();$('sp-play').textContent='暂停演示';const next=()=>{if(step>=5){stopPlayback();return;}step++;refresh();playTimer=setTimeout(next,1300);};playTimer=setTimeout(next,1300);};
      document.querySelectorAll('[data-sp-step]').forEach(b=>b.onclick=()=>{stopPlayback();step=+b.dataset.spStep;refresh();});document.querySelectorAll('[data-sp-unit]').forEach(b=>b.onclick=()=>{unit=+b.dataset.spUnit;refresh();});
    }
    $('sp-mode').onchange=()=>{trainingMode=$('sp-mode').value;refresh();};$('sp-epochs').onchange=()=>{const n=+$('sp-epochs').value;if([100,500,1000,3000,10000].includes(n))trainingEpochs=n;refresh();};
    document.querySelectorAll('[data-sp-tune]').forEach(b=>b.onclick=()=>{trainingMode=b.dataset.spTune;train(10);});
    $('sp-train-one').onclick=()=>train(1);$('sp-train').onclick=()=>train(trainingEpochs);$('sp-auto-fit').onclick=()=>train(trainingEpochs,{automatic:true});$('sp-stop').onclick=()=>{stop();if(!autoResult)status='已暂停，参数与当前位置已保留。';refresh();};$('sp-reset').onclick=()=>{stop();autoResult=null;model=createModel();ghost=null;delta=0;sampleSerial=0;status='模型已重置，十二维观测保持不变。';refresh(true);};
    if($('sp-cloud'))$('sp-cloud').onchange=()=>{showCloud=$('sp-cloud').checked;refresh();};
    document.querySelectorAll('.sp-section-tabs a').forEach(a=>a.setAttribute('aria-current',a.getAttribute('href')==='#'+view?'page':'false'));
    refresh(true);
  }
  return {template,setup,stop,refresh:()=>refresh(false),state:()=>({model,selected,view,representation,ellipsoidScope,contourScale,sampleSerial,delta,step,unit,showCloud,busy,rows:lastRows,umap:lastUmap,yaw,pitch,autoSession,autoResult,trainingEpochs,trainingMode})};
}


