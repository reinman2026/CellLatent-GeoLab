import {DynamicsLab,kineticState,kineticField,softplus,AutoFitSession} from './core.mjs';
import {ellipsoidPoint,rotateLatent,projectEllipsoid,orientLatentProjection} from './spatial-lab.mjs';
import {paperToyData} from './paper-toy-data.mjs';
import {paperToyWHistory} from './paper-toy-w-history.mjs';
import {verifiedSource} from './verified-source.mjs';

// Independent teaching model: the ODE is inside reconstruction, so every
// displayed training update differentiates through time and all RK4 stages.
export function createKineticsLab(){
  const genes=paperToyData.genes,model=new DynamicsLab({data:paperToyData.cells,genes:genes.length}),$=id=>document.getElementById(id),f=(x,n=3)=>Number(x).toFixed(n);
  const cachedW=(paperToyWHistory?.snapshots??[]).filter(s=>Array.isArray(s.W)&&s.W.length===genes.length),cacheEvery=paperToyWHistory?.snapshotEvery??10;
  const vec=x=>'['+x.map(v=>f(v)).join(', ')+']',inputDim=paperToyData.featureOrder.length;
  const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  let selected=0,gene=0,regulator=0,representation='mean',sampleSerial=0,stage=2,mode='joint',cursor=null;
  let yaw=.65,pitch=-.35,replicaYaw=.5,replicaPitch=.25,busy=false,timer=null,playTimer=null,remaining=0,before=null,lastRows=[],lastLoss=null,pointer=null,replicaPointer=null;
  let manualCount=0,lastAction='模型已初始化。先选细胞与基因，再沿六步查看计算。',autoSession=null,autoResult=null,trainingEpochs=1000,tankSnapshots=[],cacheStep=0,cacheTimer=null,cachePlaying=false,appliedCacheStep=-1;
  const stageNames=[`${inputDim}维观测`,'三维潜分布','逐基因潜时间','两池动力学',`${inputDim}维表达误差`,'反向更新'];
  const groupNames={encoder:'编码器 μ / v',time:'时间解码器',grn:'调控网络 W / b',beta:'剪接率 β',gamma:'降解率 γ'};
  const scopeNames={joint:'联合拟合 · 全部参数',rates:'只调动力学 · W / b / β / γ',...groupNames};
  function source(key){const s=verifiedSource.excerpts[key];return `<details class="sp-source"><summary>已核对源码 · ${s.path.split('/').at(-1)}:${s.start}–${s.end}</summary><pre><code>${esc(s.text)}</code></pre></details>`;}
  function tuneButton(scope,label){return `<button class="module-tune" data-kl-tune="${scope}" hidden aria-hidden="true" title="缓存回放模式下不执行实时微调">微调${label} · 10 轮</button>`;}
  function slider(id,label,min,max,step){const scope=id==='kl-beta'?'beta':id==='kl-gamma'?'gamma':['kl-bias','kl-weight'].includes(id)?'grn':id==='kl-time-bias'?'time':'encoder';return `<div class="kl-parameter-control"><label class="kl-slider" for="${id}"><span>${label}<output id="${id}-value"></output></span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}"></label>${tuneButton(scope,'此参数组')}</div>`;}
  function trainingControls(){return `<div class="fit-actions" role="group" aria-label="缓存动力学回放"><label for="kl-cache">缓存进度 <output id="kl-cache-value"></output><input id="kl-cache" type="range" min="0" max="100" step="1" value="0"></label><button id="kl-cache-play" class="primary">播放缓存路径</button><button id="kl-cache-reset">回到第 0 轮</button><span class="tag">只读 W 回放 · 每 ${cacheEvery} 轮</span></div><div hidden aria-hidden="true"><label for="kl-mode">更新范围<select id="kl-mode">${Object.entries(scopeNames).map(([m,n])=>`<option value="${m}">${n}</option>`).join('')}</select></label><label for="kl-epochs">epoch 轮数<select id="kl-epochs">${[100,500,1000,3000,10000].map(n=>`<option value="${n}">${n}</option>`).join('')}</select></label><button id="kl-step">更新 1 轮</button><button id="kl-train">按轮数拟合</button><button id="kl-auto-fit">自动优化</button><button id="kl-pause">暂停</button><button id="kl-reset">重置模型</button></div>`;}
  function template(){return `
     <section class="sp-hero kl-hero"><div><span class="sp-kicker">05 / 缓存动力学回放</span><h1>先看三个空间，再把缓存 W 放进两座蓄水池</h1><p>顶部展示第一行模型 1 的 ${inputDim} 维观测、可旋转三维潜空间和 ${inputDim} 维重建；下面按每 ${cacheEvery} 轮读取一次固定 W，观察同一条动力学轨迹怎样演变。页面不重新启动优化器。</p></div><div class="kl-hero-note"><b>同一份观测 · 固定回放</b><span>${model.data.length} 个 paper toy 细胞 / ${genes.length} 个基因<br>${inputDim} 维表达 → 3 维潜变量 → 时间 → u/s 动力学</span></div></section>
     <section id="kl-replica-preview" class="sp-panel kl-replica-preview"><div class="sp-panel-title"><div><span class="sp-kicker">第一行模型 1 · 三空间参数入口</span><h2>同一个样本的观测、潜空间与重建</h2><p>这三个画布读取当前 paper toy 数据和同一套动力学模型；中间潜空间可拖动旋转，线条只保留主轮廓和少量网格，便于观察几何结构。</p></div><button id="kl-rp-reset">复位旋转</button></div><div class="kl-replica-three"><article class="rp-chart-card"><h3>${inputDim}维观测</h3><canvas id="kl-rp-observed" width="420" height="250" aria-label="模型 1 ${inputDim}维观测"></canvas></article><article class="rp-chart-card"><h3>三维潜空间</h3><canvas id="kl-rp-latent" width="420" height="250" tabindex="0" aria-label="模型 1 可旋转三维潜空间"></canvas></article><article class="rp-chart-card"><h3>${inputDim}维重建</h3><canvas id="kl-rp-decoded" width="420" height="250" aria-label="模型 1 ${inputDim}维重建"></canvas></article></div><p class="sub">当前选中细胞为 C${String(selected+1).padStart(2,'0')}；左列和右列使用相同 ${inputDim} 个特征，中列显示同一细胞的 μ、√v 和 z。</p></section>
    <div class="sp-controls kl-controls fit-toolbar"><div class="sp-select-cell"><label for="kl-cell">细胞</label><select id="kl-cell">${model.data.map((_,i)=>`<option value="${i}">C${String(i+1).padStart(2,'0')}</option>`).join('')}</select><label for="kl-gene">基因</label><select id="kl-gene">${genes.map((g,i)=>`<option value="${i}">${g}</option>`).join('')}</select><label for="kl-repr">解码输入</label><select id="kl-repr"><option value="mean">均值 μ</option><option value="sample">一次抽样 z</option></select><button id="kl-resample">重抽样</button></div><div class="sp-view-options"><a href="#space">回看三维潜空间</a><span id="kl-epoch" class="tag"></span><span id="kl-best-step" class="tag fit-best" hidden></span>${trainingControls()}</div><div id="kl-fit-status" class="fit-status" role="status" aria-live="polite"></div></div>
    <section class="sp-panel kl-story"><div class="sp-stages">${stageNames.map((n,i)=>`<button data-kl-stage="${i}" aria-pressed="false"><b>0${i+1}</b>${n}</button>`).join('')}</div><div id="kl-stage-text" class="sp-stage-explanation" aria-live="polite"></div><div class="kl-shared-key"><span><i class="kl-dot kl-purple"></i>每细胞：μ、v、z；每细胞每基因：t</span><span><i class="kl-dot kl-teal"></i>跨细胞共享：W、b、β、γ 及网络权重</span></div></section>
    <div class="kl-triptych">
      <section id="kl-latent-panel" class="sp-panel kl-card kl-latent"><span class="sp-kicker">① 状态表示</span><h2>三维分布 → 一个输入点</h2>${tuneButton("encoder","编码器")}<canvas id="kl-latent" class="kl-canvas kl-rotate" tabindex="0" role="img" aria-label="三维潜空间：当前细胞的均值、标准差椭球和抽样点；拖动或用方向键旋转"></canvas><div id="kl-latent-values" class="kl-numbers"></div><p>椭球中心为 μ，半轴为标准差 √v。这里显示 1 倍标准差等值面，约包含 19.9% 的三维高斯概率。拖动或按方向键旋转。</p><div class="kl-formula">q(z | x) = N(μ, diag(v))<br>z = μ + √v ⊙ ε，ε ∼ N(0, I)</div></section>
      <section id="kl-time-panel" class="sp-panel kl-card"><span class="sp-kicker">② 时间解码器</span><h2>潜变量 → 各基因的时间</h2>${tuneButton("time","时间解码器")}<div id="kl-decoder-input" class="kl-input-vector"></div><div class="kl-down">↓ 3 → 8 个隐藏单元</div><div id="kl-hidden" class="kl-hidden"></div><div class="kl-down">↓ 8 → ${genes.length} 个时间输出</div><div id="kl-times" class="kl-times"></div><div class="kl-formula">hᴰ = ReLU(Aᴰz + cᴰ)<br>t<sub>ig</sub> = 4 · sigmoid(w<sub>g</sub>ᵀhᴰ + c<sub>g</sub>)</div><p id="kl-time-note"></p><p>latent time（潜时间）是模型为表达状态安排的相对位置。3 个潜坐标不对应 3 个时间轴；每个时间头都能读取完整的三维输入。</p></section>
      <section id="kl-reservoir-panel" class="sp-panel kl-card"><span class="sp-kicker">③ 连续表达变化</span><h2>转录 → 剪接 → 降解</h2>${tuneButton("rates","动力学")}<div id="kl-tanks" class="kl-tanks" role="img" aria-label="未剪接与已剪接 RNA 两池，以及当前流量"></div><div id="kl-flow-values" class="kl-numbers"></div><div class="kl-formula">α<sub>g</sub>(s) = softplus(b<sub>g</sub> + Σ<sub>j</sub> W<sub>gj</sub>s<sub>j</sub>)<br>du<sub>g</sub>/dt = α<sub>g</sub>(s) − β<sub>g</sub>u<sub>g</sub><br>ds<sub>g</sub>/dt = β<sub>g</sub>u<sub>g</sub> − γ<sub>g</sub>s<sub>g</sub></div><p>u 是 unspliced RNA（未剪接 RNA），s 是 spliced RNA（已剪接 RNA）。水位代表表达量，流量代表单位模型时间内的变化。</p></section>
    </div>
    <section class="sp-panel"><div class="sp-panel-title"><div><span class="sp-kicker">轨迹与位置分开看</span><h2>缓存 W 改变曲线；潜时间选择曲线上的位置</h2><p>实线是当前缓存 W 轨迹；圆点是当前细胞的预测；同色虚线是该细胞的观测水平。</p></div><div class="button-row"><span id="kl-time-badge" class="kl-badge"></span>${tuneButton("rates","动力学")}</div></div><canvas id="kl-trajectory" class="kl-canvas kl-trajectory" role="img" aria-label="当前基因的未剪接和已剪接表达随潜时间变化，显示拟合点、观测与操作前曲线"></canvas><div class="kl-legend"><span><i style="background:#188e91"></i>u(t) 未剪接</span><span><i style="background:#8462bd"></i>s(t) 已剪接</span><span>● 拟合值　┄ 观测值　灰色：操作前</span></div><div class="kl-scan"><label for="kl-cursor">沿曲线观察 <output id="kl-cursor-value"></output><input id="kl-cursor" type="range" min="0" max="4" step="0.01"></label><button id="kl-play">播放蓄水过程</button><button id="kl-follow">回到拟合潜时间</button></div><p id="kl-cursor-note">观察游标只控制蓄水池的查看时刻。拖动它不会更新网络、动力学参数或拟合潜时间。</p></section>
    <div class="kl-two-column">
     <section id="kl-compare-panel" class="sp-panel"><span class="sp-kicker">从动力学返回数据空间</span><h2>在各自的时间，读出 ${inputDim} 维预测</h2>${tuneButton("joint","全部参数")}<div id="kl-comparison" class="table-scroll"></div><div id="kl-cell-error" class="kl-note"></div><div class="kl-formula">û<sub>ig</sub> = u<sub>g</sub>(t<sub>ig</sub>；W, b, β, γ)<br>ŝ<sub>ig</sub> = s<sub>g</sub>(t<sub>ig</sub>；W, b, β, γ)</div><p>每个基因有自己的 t<sub>ig</sub>，因此${inputDim} 维读出会拼接每个基因求解时刻的分量。蓄水池中调控所用的 s(t<sub>ig</sub>) 来自该时刻的完整 ${genes.length} 基因 ODE（常微分方程）状态。</p></section>
    </div>
     <section id="kl-training-panel" class="sp-panel"><div class="sp-panel-title"><div><span class="sp-kicker">训练过程可视化 / 缓存 W</span><h2>同一条固定轨迹，怎样改变动力学读出？</h2><p>页面读取一次训练过程中每 ${cacheEvery} 轮保存的 W 快照，按进度回放；不在浏览器中重复计算优化过程。</p></div><span class="fit-location-note">使用顶部缓存进度回放</span></div><div id="kl-metrics" class="kl-metrics"></div><div class="kl-two-column"><div><canvas id="kl-loss" class="kl-canvas kl-loss" role="img" aria-label="各次操作后、使用固定八组独立评估抽样的目标值"></canvas><p>损失曲线来自固定训练记录；当前前向目标只用于说明所选缓存 W 对表达读出的影响。</p></div><div><div id="kl-gradients" class="table-scroll"></div><div id="kl-chain-rule" class="kl-note"></div></div></div><div class="sp-panel kl-card"><h2>缓存 W 驱动的动力学轨迹：每 10 轮记录一次蓄水池</h2><canvas id="kl-tank-history" class="kl-canvas kl-trajectory" role="img" aria-label="每十轮缓存 W 对未剪接和已剪接水池的影响"></canvas><p>曲线读取第 0 轮、每 10 轮和最终轮的 W 快照；页面不执行新的梯度更新。u 是未剪接 RNA（unspliced RNA，未剪接 RNA），s 是已剪接 RNA（spliced RNA，已剪接 RNA）。</p></div><div id="kl-status" class="kl-status" role="status" aria-live="polite"></div><div class="kl-formula">L = mean<sub>i</sub> Σ<sub>g</sub> [(û<sub>ig</sub> − u<sub>ig</sub>)² + (ŝ<sub>ig</sub> − s<sub>ig</sub>)²] / (2σ²) + 常数 + 0.05 · KL + 0.001 Σ W²<br>∂L/∂θ<sub>time</sub> = Σ<sub>i,g</sub> (∂L/∂t<sub>ig</sub>) · (∂t<sub>ig</sub>/∂θ<sub>time</sub>)</div><p>当前 W 来自固定缓存；b、β、γ 和时间头保持教学模型的固定参数。切换细胞、基因或均值 / 抽样只改变前向预览。</p></section>
    <section class="sp-panel kl-boundary"><span class="sp-kicker">与 RegVelo 对齐，以及本实验的边界</span><h2>读取缓存，回放动力学变化</h2><p>本页读取一次固定训练中的 W 快照，观察调控网络改变时两池动力学读出怎样变化。浏览器不再执行优化更新。</p><p>GRN（gene regulatory network，基因调控网络）在本页是 ${genes.length}×${genes.length} 的 W 和 ${genes.length} 个基础项 b。初始状态固定 u(0) = s(0) = 0，时间上限固定为 4；这是教学选择。潜时间单位没有实验时钟校准，β、γ 的绝对生物学单位也不能由此确定。W 来自缓存轨迹，其他动力学参数是固定教学值。</p><p>本页沿用前面 ${model.data.length} 个 ${inputDim} 维观测，使用独立的 ${inputDim} → 8 → 3 编码器、8 单元时间解码器、${genes.length} 基因耦合方程与 12 步 RK4（四阶 Runge–Kutta 数值积分）。前面页面保留直接重建模型。本页实现了“z → 时间 → ODE → 误差”的可微分连接；省略真实 RegVelo 的批次归一化、随机失活、可学习观测噪声、图先验及额外速度/转录约束等配置，也未复现其完整求解器。</p><p>最终参数与表达平均：β、γ 以及调控网络 W / b 是跨细胞共享的点估计。多次抽样共用这些参数，再分别解码时间、解方程；重构 u/s 可以沿重复抽样这一维取平均。源码表达接口 n_samples 默认 1，return_mean 默认 True，只有调大 n_samples 才能平均多个样本。当前上方重构对照保持所选 μ / z 的单路径预览，不把它标成多抽样后验平均。</p>${source('heads')}${source('time')}${source('ode')}${source('loss')}${source('global_rates')}${source('posterior_api')}${source('posterior_fit')}<a class="kl-next" href="#construction">继续查看原始源码对应 →</a></section>`;}

  function rows(){return model.data.map((_,i)=>model.trace(i,{useMean:representation==='mean',serial:sampleSerial}));}
  function curves(parameters=model.p){return Array.from({length:81},(_,i)=>{const time=model.tMax*i/80,y=kineticState(time,parameters,model.odeSteps,genes.length);return {time,u:y[gene],s:y[genes.length+gene]};});}
  function capture(){const r=model.trace(selected,{useMean:representation==='mean',serial:sampleSerial});before={curve:curves(),time:r.time[gene],gene,selected};}
  function metrics(){lastLoss=model.assess();return lastLoss;}
  function setCachedW(W){for(let g=0;g<genes.length;g++)for(let j=0;j<genes.length;j++)model.p[`kw${g}_${j}`]=Number(W?.[g]?.[j]??0);}
  function snapshotAt(){return cachedW[Math.min(cachedW.length-1,Math.max(0,Math.round(cacheStep/100*Math.max(0,cachedW.length-1))))]??null;}
  function rebuildCachedTankHistory(){
    if(!cachedW.length){tankSnapshots=[];return;}
    const saved=genes.flatMap((_,g)=>genes.map((_,j)=>[`kw${g}_${j}`,model.p[`kw${g}_${j}`]]));
    tankSnapshots=cachedW.map(s=>{setCachedW(s.W);const trace=model.trace(selected,{useMean:representation==='mean',serial:sampleSerial});return {epoch:s.epoch,cell:selected,gene,representation,u:trace.prediction[genes.length+gene],s:trace.prediction[gene],time:trace.time[gene]};});
    saved.forEach(([key,value])=>{model.p[key]=value;});
  }
  function applyCachedW(force=false){const snap=snapshotAt();if(!snap)return;if(!force&&appliedCacheStep===cacheStep)return;setCachedW(snap.W);model.epoch=snap.epoch;model.last=null;metrics();lastRows=rows();appliedCacheStep=cacheStep;lastAction=`只读回放第 ${snap.epoch} 轮缓存 W；没有重新训练。`;}
  function stopCache(){if(cacheTimer!==null)clearTimeout(cacheTimer);cacheTimer=null;cachePlaying=false;if($('kl-cache-play'))$('kl-cache-play').textContent='播放缓存路径';}
  function playCache(){stopCache();cachePlaying=true;$('kl-cache-play').textContent='暂停';const tick=()=>{if(!cachePlaying)return;if(cacheStep>=100){stopCache();return;}cacheStep=Math.min(100,cacheStep+100/Math.max(1,cachedW.length-1));applyCachedW();refresh();cacheTimer=setTimeout(tick,180);};tick();}
  function stageText(r){const g=gene,n=genes[g];return [
     `C${String(selected+1).padStart(2,'0')} 的输入为 [${paperToyData.featureOrder.join(', ')}] = ${vec(r.raw)}。模型只读取这些表达，不读取已知发育时间。`,
    `编码器得到三维均值 μ = ${vec(r.mu)} 和方差 v = ${vec(r.variance)}。当前用${representation==='mean'?'均值 μ':'样本 z'} ${vec(r.latent)} 进入时间解码器。`,
     `同一个三维输入经过共享隐藏层，再分别得到 ${genes.map(x=>`t_${x}`).join('、')}。当前基因 ${n} 的时间为 ${f(r.time[g])} / 4。时间头将所有潜坐标结合起来，不把某个 z 轴直接当作时钟。`,
    `W、b 先将当前完整 s 状态变成转录流入 α(s)。β 使 u 转入 s，γ 使 s 流出。解方程得到整条轨迹，再在 t_${n} = ${f(r.time[g])} 读取基因 ${n}。`,
     `基因 ${n}：预测 (u, s) = (${f(r.prediction[genes.length+g])}, ${f(r.prediction[g])})，观测为 (${f(r.raw[genes.length+g])}, ${f(r.raw[g])})。${genes.length} 个基因合成 ${inputDim} 维预测，所有细胞一起形成训练目标。`,
     `当前显示第 ${model.epoch} 轮缓存 W。页面只读取这条固定轨迹，不执行新的梯度更新；b、β、γ 和时间头保持教学模型的固定参数。`
  ][stage];}
  function panelHighlights(){document.querySelectorAll('[data-kl-stage]').forEach(b=>{const on=+b.dataset.klStage===stage;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));});
    for(const [id,stages] of [['kl-latent-panel',[0,1]],['kl-time-panel',[2]],['kl-reservoir-panel',[3]],['kl-compare-panel',[4]],['kl-training-panel',[5]]])$(id).classList.toggle('kl-highlight',stages.includes(stage));}
  function initCanvas(id){const c=$(id),w=Math.max(160,c.clientWidth),h=c.clientHeight||280,d=window.devicePixelRatio||1;c.width=Math.round(w*d);c.height=Math.round(h*d);const g=c.getContext('2d');g.scale(d,d);g.font='11px "Microsoft YaHei",sans-serif';return {c,g,w,h};}
  function drawReplicaExpression(id,values,observed){const {g,w,h}=initCanvas(id),max=Math.max(1,...model.data.flatMap(c=>c.raw),...values),x=j=>35+j*(w-60)/Math.max(1,inputDim-1),y=v=>h-35-v*(h-65)/max;g.strokeStyle='#dbe4ed';g.fillStyle='#53667b';for(let j=0;j<inputDim;j++){g.beginPath();g.moveTo(x(j),20);g.lineTo(x(j),h-30);g.stroke();g.fillText(paperToyData.featureOrder[j],x(j)-12,h-12);}const line=(v,color,width,dash=[])=>{g.strokeStyle=color;g.lineWidth=width;g.setLineDash(dash);g.beginPath();v.forEach((z,j)=>j?g.lineTo(x(j),y(z)):g.moveTo(x(j),y(z)));g.stroke();g.setLineDash([]);};model.data.forEach((c,i)=>{if(i!==selected)line(c.raw,'#c1ceda',.8);});if(observed)line(observed,'#8b6b49',2,[5,4]);line(values,'#1568a0',3);}
  function drawReplicaLatent(){const {g,w,h}=initCanvas('kl-rp-latent'),rs=lastRows.length?lastRows:model.data.map((_,i)=>model.trace(i,{useMean:true,serial:sampleSerial})),points=rs.map(r=>r.mu),frame=orientLatentProjection(points[0],rs[0].variance,replicaYaw,replicaPitch),span=Math.max(1,...points.flat().map(Math.abs),...rs.flatMap(r=>r.mu.map((v,j)=>Math.abs(v)+2*Math.sqrt(r.variance[j])))),scale=Math.min(w,h)*.37/span,project=v=>{const r=rotateLatent(v,replicaYaw,replicaPitch),q=frame.rotate(r);return [w/2+q[0]*scale,h/2-q[1]*scale,r[2]]};g.font='11px sans-serif';for(let j=0;j<3;j++){const v=[0,0,0];v[j]=span;const q=project(v);g.strokeStyle='#bcc8d5';g.beginPath();g.moveTo(w/2,h/2);g.lineTo(q[0],q[1]);g.stroke();g.fillStyle='#53667b';g.fillText(`z${j+1}`,q[0],q[1]);}const ordered=points.map((v,i)=>({v,i})).sort((a,b)=>project(a.v)[2]-project(b.v)[2]);ordered.forEach(({v,i})=>{const q=project(v);g.fillStyle=i===selected?'#1568a0':'#9ab6c6';g.beginPath();g.arc(q[0],q[1],i===selected?5:2.5,0,Math.PI*2);g.fill();});const mu=rs[selected].mu,shape=projectEllipsoid(mu,rs[selected].variance,1,replicaYaw,replicaPitch),center=project(mu);g.save();g.translate(center[0],center[1]);g.rotate(0);g.fillStyle='#247ba018';g.strokeStyle='#1568a0';g.lineWidth=1.8;g.beginPath();g.ellipse(0,0,shape.radii[0]*scale,shape.radii[1]*scale,0,0,Math.PI*2);g.fill();g.stroke();g.restore();g.fillStyle='#687b8c';g.fillText('μ / √v / z；拖动旋转',12,h-12);}
  function drawReplicaPreview(){const r=lastRows[selected]??model.trace(selected,{useMean:representation==='mean',serial:sampleSerial});drawReplicaExpression('kl-rp-observed',model.data[selected].raw,null);drawReplicaExpression('kl-rp-decoded',r.prediction,model.data[selected].raw);drawReplicaLatent();}
  function drawLatent(rs){const {g,w,h}=initCanvas('kl-latent'),r=rs[selected],projected=rs.map(x=>rotateLatent(x.mu,yaw,pitch));
    const shape=projectEllipsoid(r.mu,r.variance,1,yaw,pitch),sample=rotateLatent(r.latent,yaw,pitch);
    const span=Math.max(1.2,...projected.flatMap(p=>[Math.abs(p[0]),Math.abs(p[1])]),Math.abs(shape.center[0])+shape.extent[0],Math.abs(shape.center[1])+shape.extent[1],Math.abs(sample[0]),Math.abs(sample[1]));
    const scale=Math.min(w-60,h-65)/(2*span),screen=p=>[w/2+p[0]*scale,h/2-p[1]*scale];
    const line=(points,color,dash=[])=>{g.beginPath();g.strokeStyle=color;g.setLineDash(dash);points.forEach((p,j)=>{const q=screen(p);j?g.lineTo(...q):g.moveTo(...q);});g.stroke();g.setLineDash([]);};
    for(let j=0;j<3;j++){const axis=[0,0,0];axis[j]=span*.85;const p=rotateLatent(axis,yaw,pitch);line([[0,0,0],p],['#85b9b4','#ab98d0','#c1a38a'][j]);const q=screen(p);g.fillStyle='#788899';g.fillText('z'+(j+1),q[0]+3,q[1]-4);}
    const [cx,cy]=screen(shape.center);g.save();g.translate(cx,cy);g.rotate(shape.angle);g.fillStyle='#8660bb19';g.strokeStyle='#9a7ac9';g.beginPath();g.ellipse(0,0,shape.radii[0]*scale,shape.radii[1]*scale,0,0,2*Math.PI);g.fill();g.stroke();g.restore();
    // Wireframe uses the same 3D radii and rotation as the projected silhouette.
    for(const lat of [-.65,0,.65])line(Array.from({length:49},(_,j)=>rotateLatent(ellipsoidPoint(r.mu,r.variance,1,lat,j/48*2*Math.PI),yaw,pitch)),'#a78cc462');
    for(const lon of [0,Math.PI/3,2*Math.PI/3])line(Array.from({length:49},(_,j)=>rotateLatent(ellipsoidPoint(r.mu,r.variance,1,j/48*2*Math.PI,lon),yaw,pitch)),'#a78cc462');
    projected.map((p,i)=>({p,i})).sort((a,b)=>a.p[2]-b.p[2]).forEach(({p,i})=>{const [x,y]=screen(p);g.fillStyle=i===selected?'#8360b4':'#9ab6c6';g.globalAlpha=i===selected?1:.65;g.beginPath();g.arc(x,y,i===selected?5:3,0,Math.PI*2);g.fill();});g.globalAlpha=1;
    if(representation==='sample'){line([shape.center,sample],'#cf9a33',[3,3]);const [x,y]=screen(sample);g.fillStyle='#c48b22';g.beginPath();g.arc(x,y,5,0,Math.PI*2);g.fill();g.fillText('z',x+8,y-6);}
    g.fillStyle='#745994';g.fillText('μ',cx+8,cy-7);g.fillStyle='#8b98a3';g.fillText('${model.data.length} 个细胞均值 · 选中细胞 1σ 椭球',12,h-12);
  }
  function drawTanks(r){const time=cursor??r.time[gene],y=kineticState(time,model.p,model.odeSteps,genes.length),field=kineticField(y,model.p,genes.length),p=model.rates()[gene],u=y[gene],s=y[gene+genes.length],alpha=softplus(p.bias+p.weights.reduce((sum,v,j)=>sum+v*y[genes.length+j],0));
    const bound=Math.max(1,...curves().flatMap(q=>[q.u,q.s])),height=v=>Math.max(0,Math.min(110,110*v/bound));
    const a=height(u),b=height(s),arrow=(x1,y1,x2,y2,color)=>`<path d="M ${x1} ${y1} L ${x2} ${y2}" stroke="${color}" stroke-width="3" fill="none"/><path d="M ${x2-6} ${y2-5} L ${x2} ${y2} L ${x2-6} ${y2+5}" stroke="${color}" stroke-width="2" fill="none"/>`;
    $('kl-tanks').innerHTML=`<svg viewBox="0 0 360 242" role="img" aria-label="基因 ${genes[gene]}，u 水位 ${f(u)}，s 水位 ${f(s)}"><text x="180" y="20" text-anchor="middle" fill="#687b8a" font-size="12">${cursor===null?'拟合时刻':'观察游标'} t = ${f(time)}</text><text x="47" y="55" text-anchor="middle" fill="#188e91" font-size="12">转录</text>${arrow(16,73,70,73,'#188e91')}<rect x="72" y="${195-a}" width="90" height="${a}" rx="4" fill="#b4deda"/><path d="M72 65 V195 H162 V65" stroke="#188e91" fill="none" stroke-width="2"/><text x="117" y="${Math.max(85,186-a)}" text-anchor="middle" fill="#157f84" font-size="16">u ${f(u,2)}</text>${arrow(163,123,211,123,'#95a1b0')}<text x="188" y="109" text-anchor="middle" fill="#7c8997" font-size="11">剪接</text><rect x="214" y="${195-b}" width="90" height="${b}" rx="4" fill="#d1c4e7"/><path d="M214 65 V195 H304 V65" stroke="#8462bd" fill="none" stroke-width="2"/><text x="259" y="${Math.max(85,186-b)}" text-anchor="middle" fill="#7959aa" font-size="16">s ${f(s,2)}</text>${arrow(306,164,349,164,'#8462bd')}<text x="328" y="149" text-anchor="middle" fill="#8462bd" font-size="11">降解</text><text x="117" y="220" text-anchor="middle" fill="#6c8792" font-size="11">未剪接 RNA</text><text x="259" y="220" text-anchor="middle" fill="#6c8792" font-size="11">已剪接 RNA</text></svg>`;
     $('kl-flow-values').innerHTML=`<span>流入 α(s)<b>${f(alpha)}</b></span><span>剪接 βu<b>${f(p.beta*u)}</b></span><span>流出 γs<b>${f(p.gamma*s)}</b></span><span>du/dt<b>${f(field[gene])}</b></span><span>ds/dt<b>${f(field[gene+genes.length])}</b></span><span>水位参考上限<b>${f(bound,2)}</b></span>`;
  }
  function plot(id,series,{maxX=4,minY=0,maxY=1,marker=null,observed=[],oldMarker=null,cursorAt=null,xLabel='相对潜时间 t',emptyText=''}={}){
    const {g,w,h}=initCanvas(id),left=43,right=w-18,top=18,bottom=h-35,span=Math.max(1e-5,maxY-minY),x=v=>left+(right-left)*v/Math.max(maxX,1),y=v=>bottom-(bottom-top)*(v-minY)/span;
    g.lineWidth=1;for(let k=0;k<=4;k++){const v=minY+span*k/4;g.strokeStyle='#e8edf1';g.beginPath();g.moveTo(left,y(v));g.lineTo(right,y(v));g.stroke();g.fillStyle='#8a98a3';g.textAlign='right';g.fillText(f(v,1),left-6,y(v)+3);}
    for(let k=0;k<=4;k++){g.fillStyle='#8a98a3';g.textAlign='center';g.fillText(f(maxX*k/4,maxX>10?0:1),x(maxX*k/4),bottom+16);}g.fillText(xLabel,(left+right)/2,h-3);g.textAlign='left';
    for(const item of observed){g.setLineDash([5,5]);g.strokeStyle=item.color+'80';g.beginPath();g.moveTo(left,y(item.value));g.lineTo(right,y(item.value));g.stroke();}g.setLineDash([]);
    for(const item of series){g.strokeStyle=item.color;g.lineWidth=item.old?1:2.2;g.setLineDash(item.old?[3,4]:[]);g.beginPath();item.points.forEach((p,i)=>i?g.lineTo(x(p[0]),y(p[1])):g.moveTo(x(p[0]),y(p[1])));g.stroke();g.setLineDash([]);}
    for(const [t,color,dash] of [[oldMarker,'#adb8c2',[3,5]],[marker,'#cf9b3d',[]],[cursorAt,'#6c8da7',[2,3]]])if(t!==null&&t!==undefined){g.strokeStyle=color;g.lineWidth=1;g.setLineDash(dash);g.beginPath();g.moveTo(x(t),top);g.lineTo(x(t),bottom);g.stroke();g.setLineDash([]);}
    if(marker!==null)for(const item of series.filter(s=>!s.old)){const at=model.trace(selected,{useMean:representation==='mean',serial:sampleSerial}),v=item.kind==='u'?at.prediction[genes.length+gene]:at.prediction[gene];g.fillStyle=item.color;g.beginPath();g.arc(x(marker),y(v),4.5,0,2*Math.PI);g.fill();}
    if(emptyText){g.fillStyle='#82929d';g.fillText(emptyText,left+12,top+25);}
  }
  function drawTrajectory(r){const curve=curves(),old=before?.gene===gene&&before?.selected===selected?before:null;
    const series=[...(old?['u','s'].map(kind=>({old:true,color:'#b3bdc8',points:old.curve.map(p=>[p.time,p[kind]])})):[]),...['u','s'].map(kind=>({kind,color:kind==='u'?'#188e91':'#8462bd',points:curve.map(p=>[p.time,p[kind]])}))];
    const maxY=Math.max(1,...series.flatMap(s=>s.points.map(p=>p[1])),r.raw[gene],r.raw[gene+genes.length])*1.13;
    plot('kl-trajectory',series,{maxY,marker:r.time[gene],oldMarker:old?.time,cursorAt:cursor,observed:[{color:'#188e91',value:r.raw[gene+genes.length]},{color:'#8462bd',value:r.raw[gene]}]});
  }
  function drawLoss(){const h=paperToyWHistory?.history??model.history,points=h.map((r,i)=>[r.epoch??i+1,r.validation_loss??r.loss??r.train_loss??0]),all=points.map(p=>p[1]),min=Math.min(...all,0),max=Math.max(...all,1);plot('kl-loss',[{color:'#188e91',points}],{maxX:Math.max(1,points.at(-1)?.[0]??1),minY:min-(max-min)*.08,maxY:max+(max-min)*.1,xLabel:'训练轮次 · 缓存记录',emptyText:points.length<2?'没有缓存损失记录':''});}
  function drawTankHistory(){
    const {g,w,h}=initCanvas('kl-tank-history'),left=43,right=w-18,top=18,bottom=h-35;
    const maxX=Math.max(10,...tankSnapshots.map(s=>s.epoch)),maxY=Math.max(1,...tankSnapshots.flatMap(s=>[s.u,s.s]))*1.12,span=maxY;
    const x=v=>left+(right-left)*v/Math.max(maxX,1),y=v=>bottom-(bottom-top)*v/span;
    g.lineWidth=1;g.textAlign='right';
    for(let k=0;k<=4;k++){const v=span*k/4;g.strokeStyle='#e8edf1';g.beginPath();g.moveTo(left,y(v));g.lineTo(right,y(v));g.stroke();g.fillStyle='#8a98a3';g.fillText(f(v,1),left-6,y(v)+3);}
    g.textAlign='center';for(let k=0;k<=4;k++){const v=maxX*k/4;g.fillStyle='#8a98a3';g.fillText(f(v,maxX>10?0:1),x(v),bottom+16);}g.fillText('训练更新次数',(left+right)/2,h-3);g.textAlign='left';
    if(!tankSnapshots.length){g.fillStyle='#82929d';g.fillText('读取缓存 W 后显示动力学轨迹',left+12,top+25);return;}
    for(const [key,color,label] of [['u','#188e91','u 未剪接'],['s','#8462bd','s 已剪接']]){
      g.strokeStyle=color;g.lineWidth=2.2;g.beginPath();tankSnapshots.forEach((item,i)=>{const q=[x(item.epoch),y(item[key])];i?g.lineTo(...q):g.moveTo(...q);});g.stroke();
      g.fillStyle=color;tankSnapshots.forEach(item=>{g.beginPath();g.arc(x(item.epoch),y(item[key]),4,0,2*Math.PI);g.fill();});
      const last=tankSnapshots.at(-1);g.fillText(label,x(last.epoch)+7,y(last[key])-5);
    }
  }
  function syncSlider(id,value){const input=$(id);input.min=Math.min(+input.getAttribute('min'),value);input.max=Math.max(+input.getAttribute('max'),value);input.value=value;$(id+'-value').textContent=f(value);}
  function updateText(r){
    $('kl-stage-text').textContent=stageText(r);panelHighlights();
    $('kl-latent-values').innerHTML=`<span>均值 μ<b>${vec(r.mu)}</b></span><span>标准差 √v<b>${vec(r.variance.map(Math.sqrt))}</b></span><span>实际解码输入<b>${vec(r.latent)}</b></span>`;
    $('kl-decoder-input').textContent=(representation==='mean'?'μ = ':'z = ')+vec(r.latent);
    $('kl-hidden').innerHTML=r.decoderHidden.map((v,j)=>`<span title="隐藏单元 ${j+1}"><small>h${j+1}</small><b>${f(v,2)}</b></span>`).join('');
    $('kl-times').innerHTML=r.time.map((t,g)=>`<button data-kl-gene="${g}" class="${gene===g?'active':''}" aria-pressed="${gene===g}"><span>基因 ${genes[g]}<b>${f(t)}</b></span><i style="width:${t/4*100}%"></i></button>`).join('');
    document.querySelectorAll('[data-kl-gene]').forEach(b=>b.onclick=()=>{stop();gene=+b.dataset.klGene;cursor=null;before=null;$('kl-gene').value=gene;refresh();});
    $('kl-time-note').textContent=`当前 ${genes[gene]} 头：${f(model.p['tmb'+gene])} + Σ wⱼhⱼ = ${f(r.timePre[gene])}。sigmoid 将它压入 0～1：${f(r.time[gene]/4)}；乘 4 得到时间 ${f(r.time[gene])}。`;
    $('kl-time-badge').textContent=`基因 ${genes[gene]} · t = ${f(r.time[gene])}`;
    $('kl-cursor').value=cursor??r.time[gene];$('kl-cursor-value').textContent=f(cursor??r.time[gene]);
    $('kl-cursor-note').textContent=cursor===null?'蓄水池当前跟随拟合潜时间。拖动游标可查看其他时刻，模型参数保持当前值。':`蓄水池正在查看 t = ${f(cursor)}；金色拟合线仍在 t = ${f(r.time[gene])}。观察游标不参与损失。`;
    const p=model.rates()[gene];
    const fullState=r.integratedStates[gene],terms=p.weights.map((w,j)=>w*fullState[genes.length+j]);
    $('kl-comparison').innerHTML=`<table><thead><tr><th>分量</th><th>观测</th><th>预测</th><th>残差</th></tr></thead><tbody>${r.raw.map((v,j)=>`<tr class="${j%genes.length===gene?'kl-selected-row':''}"><td>${j<inputDim/2?'S':'U'}_${genes[j%genes.length]}</td><td>${f(v)}</td><td>${f(r.prediction[j])}</td><td>${f(r.prediction[j]-v)}</td></tr>`).join('')}</tbody></table>`;
    $('kl-cell-error').textContent=`当前预览细胞 ${inputDim} 维平方误差 = ${f(r.raw.reduce((s,v,j)=>s+(v-r.prediction[j])**2,0),4)}。模型评估覆盖全部 ${model.data.length} 个细胞和 8 组独立抽样。本页观测噪声标准差 σ = 0.2，保持固定；它与潜分布方差 v 是两类不同的量。`;
    const loss=lastLoss??metrics();$('kl-metrics').innerHTML=[['缓存轮次',model.epoch],['当前前向目标 L',f(loss.loss,4)],['重建项',f(loss.reconstruction,4)],['KL × 0.05',f(loss.kl*.05,4)],['W² 正则',f(loss.regulation,5)]].map(([label,value])=>`<div><span>${label}</span><b>${value}</b></div>`).join('');
    const groupTable=`<table><thead><tr><th>参数组</th><th>梯度范数</th><th>实际更新量</th></tr></thead><tbody>${Object.entries(groupNames).map(([key,label])=>{const g=model.last?.groups[key];return `<tr><td>${label}${g&&!g.enabled?' · 冻结':''}</td><td>${g?f(g.gradientNorm,4):'待更新'}</td><td>${g?f(g.updateNorm,4):'待更新'}</td></tr>`;}).join('')}</tbody></table><p>当前为只读缓存回放，没有新的梯度或参数更新。</p>`;
    const parameterTable=`<h3>基因 ${genes[gene]} · 当前缓存参数</h3><table><thead><tr><th>参数</th><th>更新前</th><th>更新后</th></tr></thead><tbody>${[['β','kbeta'+gene,true],['γ','kgamma'+gene,true],['b','kb'+gene,false],['时间偏置 c','tmb'+gene,false]].map(([name,key,positive])=>{const change=model.last?.changes[key],transform=positive?softplus:x=>x;return `<tr><td>${name}</td><td>${change?f(transform(change.before),4):'待更新'}</td><td>${f(transform(model.p[key]),4)}</td></tr>`;}).join('')}</tbody></table><p>β、γ 由固定教学参数显示；本页只替换缓存 W。</p>`;
    $('kl-gradients').innerHTML=groupTable+parameterTable;
    const y=r.integratedStates[gene],v=kineticField(y,model.p,genes.length),dldt=((r.prediction[genes.length+gene]-r.raw[genes.length+gene])*v[gene]+(r.prediction[gene]-r.raw[gene])*v[genes.length+gene])/(model.sigma**2);
    $('kl-chain-rule').textContent=`当前预览细胞、基因 ${genes[gene]} 的局部重建斜率 ∂ℓ/∂t ≈ ${f(dldt,4)}。它由“u 残差 × du/dt + s 残差 × ds/dt”除以 σ² 得到，用于解释当前缓存 W 的局部动力学。`;
    $('kl-status').textContent=(busy?`正在拟合，还剩 ${remaining} 步。`:'')+lastAction;
    $('kl-epoch').textContent='缓存第 '+model.epoch+' 轮';
    $('kl-best-step').hidden=!autoSession&&!autoResult;$('kl-best-step').textContent=autoSession?`本轮最佳 第 ${autoSession.best.epoch-autoSession.startEpoch} 步`:autoResult?`本轮最佳 第 ${autoResult.bestStep} 步`:'';
    $('kl-fit-status').textContent=cachedW.length?`只读缓存回放 · 当前第 ${model.epoch} 轮；页面不执行新的优化更新。`:autoSession?`自动优化中 · 已计算 ${autoSession.attempts}/${autoSession.options.maxSteps} 轮 · 本轮最佳第 ${autoSession.best.epoch-autoSession.startEpoch} 轮。可暂停并保留最佳参数。`:autoResult?autoSummary():busy?`正在拟合，还剩 ${remaining} 轮。`:lastAction;
    $('kl-epochs').value=trainingEpochs;
    for(const id of ['kl-step','kl-train','kl-auto-fit','kl-epochs','kl-mode'])$(id).disabled=busy||cachedW.length>0;
    document.querySelectorAll('[data-kl-tune]').forEach(b=>b.disabled=busy);
    $('kl-pause').disabled=!busy;
  }
  function refresh(recompute=false){if(!$('kl-latent'))return;if(cachedW.length)applyCachedW();if(recompute)metrics();if($('kl-cache')){$('kl-cache').value=cacheStep;$('kl-cache-value').textContent=`第 ${snapshotAt()?.epoch??0} 轮 · ${Math.min(cachedW.length,Math.round(cacheStep/100*Math.max(0,cachedW.length-1))+1)}/${Math.max(1,cachedW.length)}`;}lastRows=rows();const r=lastRows[selected];updateText(r);drawReplicaPreview();drawLatent(lastRows);drawTanks(r);drawTrajectory(r);drawLoss();drawTankHistory();}
  function stopPlay(){clearTimeout(playTimer);playTimer=null;if($('kl-play'))$('kl-play').textContent='播放蓄水过程';}
  function autoSummary(){const r=autoResult,reason=r.reason==='plateau'?'改善趋缓，已停止':r.reason==='limit'?'已达到计算上限':r.reason==='error'?'出现异常，已停止':'已暂停';return `${reason} · 本轮最佳：第 ${r.bestStep} 步（累计第 ${r.bestEpoch} 次）· 实际计算 ${r.attempts} 步 · L ${f(r.startLoss,4)} → ${f(r.bestLoss,4)}。已保留最佳参数；仅依据当前教学数据的训练目标。`;}
  function finishAuto(reason){if(!autoSession)return;capture();autoResult=autoSession.finish(model,reason);autoSession=null;metrics();lastAction=autoSummary();}
  function stop(){stopCache();if(autoSession)finishAuto('paused');clearTimeout(timer);timer=null;busy=false;remaining=0;stopPlay();}
  function recordTankSnapshot(force=false){
    const epoch=model.epoch;
    if(!force&&epoch!==0&&epoch%10!==0)return;
    const trace=model.trace(selected,{useMean:representation==='mean',serial:sampleSerial});
    const snapshot={epoch,cell:selected,gene,representation,u:trace.prediction[genes.length+gene],s:trace.prediction[gene],time:trace.time[gene]};
    const index=tankSnapshots.findIndex(item=>item.epoch===epoch);
    if(index>=0)tankSnapshots[index]=snapshot;else tankSnapshots.push(snapshot);
    tankSnapshots.sort((a,b)=>a.epoch-b.epoch);
  }
  function resetTankSnapshots(){tankSnapshots=[];if(cachedW.length)rebuildCachedTankHistory();else recordTankSnapshot(true);}
  function singleUpdate(){capture();const initial=lastLoss??metrics();model.step({mode});metrics();Object.assign(model.history.at(-1),lastLoss);lastAction=`${scopeNames[mode]} · 8 组抽样评估：L ${f(initial.loss,4)} → ${f(lastLoss.loss,4)}。梯度使用固定 ${model.gradientBatchSize} 个样本的小批量；只更新选定参数组。`;}
  function run(){if(!busy)return;try{const steps=Math.min(2,remaining);for(let k=0;k<steps&&remaining>0;k++){singleUpdate();remaining--;recordTankSnapshot(remaining<=0);if(autoSession?.observe(model,lastLoss)){finishAuto(autoSession.reason);recordTankSnapshot(true);busy=false;timer=null;break;}}if(remaining<=0){busy=false;timer=null;}refresh();if(busy)timer=setTimeout(run,35);}catch(error){if(autoSession)finishAuto('error');stop();lastAction=error.message+(autoResult?' '+autoSummary():'');refresh();}}
  function startTraining(automatic=false,count=trainingEpochs){stop();autoResult=null;remaining=count;autoSession=automatic?new AutoFitSession(model,metrics(),{maxSteps:remaining}):null;busy=true;refresh();timer=setTimeout(run,0);}
  function manual(key,value,label){stop();autoResult=null;if(before===null)capture();model.p[key]=value;delete model.m[key];delete model.v[key];delete model.parameterSteps[key];model.last=null;manualCount++;metrics();model.history.push({epoch:model.epoch,...lastLoss,manual:true});lastAction=`手动修改 ${label}；已重新计算所有细胞的目标值。`;refresh();}
  function setup(){
    $('kl-cell').value=selected;$('kl-gene').value=gene;$('kl-repr').value=representation;$('kl-mode').value=mode;
    $('kl-cache').oninput=()=>{stopCache();cacheStep=+$('kl-cache').value;applyCachedW();resetTankSnapshots();refresh();};
    $('kl-cache-play').onclick=()=>cachePlaying?stopCache():playCache();
    $('kl-cache-reset').onclick=()=>{stopCache();cacheStep=0;applyCachedW(true);resetTankSnapshots();refresh();};
    $('kl-cell').onchange=()=>{stop();selected=+$('kl-cell').value;before=null;cursor=null;resetTankSnapshots();refresh();};
    $('kl-gene').onchange=()=>{stop();gene=+$('kl-gene').value;before=null;cursor=null;resetTankSnapshots();refresh();};
    $('kl-repr').onchange=()=>{stop();representation=$('kl-repr').value;before=null;resetTankSnapshots();refresh();};
    $('kl-resample').onclick=()=>{stop();representation='sample';$('kl-repr').value='sample';sampleSerial++;before=null;resetTankSnapshots();refresh();};
    $('kl-mode').onchange=()=>{mode=$('kl-mode').value;refresh();};
    $('kl-epochs').onchange=()=>{const n=+$('kl-epochs').value;if([100,500,1000,3000,10000].includes(n))trainingEpochs=n;refresh();};
    document.querySelectorAll('[data-kl-tune]').forEach(b=>b.onclick=()=>{mode=b.dataset.klTune;$('kl-mode').value=mode;startTraining(false,10);});
    document.querySelectorAll('[data-kl-stage]').forEach(b=>b.onclick=()=>{stage=+b.dataset.klStage;refresh();});
    $('kl-step').onclick=()=>{stop();autoResult=null;try{singleUpdate();recordTankSnapshot(true);refresh();}catch(e){lastAction=e.message;refresh();}};
    $('kl-train').onclick=()=>startTraining();$('kl-auto-fit').onclick=()=>startTraining(true);
    $('kl-pause').onclick=()=>{stop();if(!autoResult)lastAction='拟合已暂停，保留当前参数。';refresh();};
    $('kl-reset').onclick=()=>{stop();autoResult=null;model.reset();cursor=null;before=null;manualCount=0;sampleSerial=0;tankSnapshots=[];lastAction='本页模型已重置；可以重新比较三种拟合范围。';metrics();model.history.push({epoch:0,...lastLoss});recordTankSnapshot(true);refresh();};
    $('kl-cursor').oninput=()=>{stopPlay();cursor=+$('kl-cursor').value;refresh();};
    $('kl-follow').onclick=()=>{stopPlay();cursor=null;refresh();};
    $('kl-play').onclick=()=>{if(playTimer!==null){stopPlay();return;}cursor=cursor===null||cursor>=4?0:cursor;const tick=()=>{cursor=Math.min(4,cursor+.04);refresh();if(cursor<4){$('kl-play').textContent='暂停播放';playTimer=setTimeout(tick,80);}else stopPlay();};playTimer=setTimeout(tick,0);};
    const c=$('kl-latent');c.onpointerdown=e=>{pointer={x:e.clientX,y:e.clientY};c.setPointerCapture?.(e.pointerId);};c.onpointermove=e=>{if(!pointer)return;yaw+=(e.clientX-pointer.x)*.004;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-pointer.y)*.004));pointer={x:e.clientX,y:e.clientY};drawLatent(lastRows);};c.onpointerup=c.onpointercancel=()=>{pointer=null;};
    c.onkeydown=e=>{const moves={ArrowLeft:[-.06,0],ArrowRight:[.06,0],ArrowUp:[0,-.06],ArrowDown:[0,.06]};if(moves[e.key]){e.preventDefault();yaw+=moves[e.key][0];pitch=Math.max(-1.4,Math.min(1.4,pitch+moves[e.key][1]));drawLatent(lastRows);}};
    const rc=$('kl-rp-latent');rc.onpointerdown=e=>{replicaPointer={x:e.clientX,y:e.clientY};rc.setPointerCapture?.(e.pointerId);};rc.onpointermove=e=>{if(!replicaPointer)return;replicaYaw+=(e.clientX-replicaPointer.x)*.0035;replicaPitch=Math.max(-1.4,Math.min(1.4,replicaPitch+(e.clientY-replicaPointer.y)*.0035));replicaPointer={x:e.clientX,y:e.clientY};drawReplicaLatent();};rc.onpointerup=rc.onpointercancel=()=>{replicaPointer=null;};rc.onkeydown=e=>{const moves={ArrowLeft:[-.08,0],ArrowRight:[.08,0],ArrowUp:[0,-.08],ArrowDown:[0,.08]};if(moves[e.key]){e.preventDefault();replicaYaw+=moves[e.key][0];replicaPitch=Math.max(-1.4,Math.min(1.4,replicaPitch+moves[e.key][1]));drawReplicaLatent();}};$('kl-rp-reset').onclick=()=>{replicaYaw=.5;replicaPitch=.25;drawReplicaLatent();};
    stopCache();cacheStep=0;applyCachedW(true);rebuildCachedTankHistory();if(!model.history.length){model.history.push({epoch:model.epoch,...lastLoss});}refresh();
  }
  return {template,setup,stop,refresh,state:()=>({model,selected,gene,regulator,representation,sampleSerial,stage,mode,cursor,busy,remaining,rows:lastRows,loss:lastLoss,before,yaw,pitch,manualCount,autoSession,autoResult,trainingEpochs,tankSnapshots})};
}





