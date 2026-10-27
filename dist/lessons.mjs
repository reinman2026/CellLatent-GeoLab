import {NetworkLab,RNG} from './core.mjs';
import {verifiedSource} from './verified-source.mjs';

export function createLessons(sharedModel){
  const $=id=>document.getElementById(id),f=(n,d=3)=>Number(n).toFixed(d);
  const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const vector=v=>'['+v.map(x=>f(x)).join(', ')+']';
  const colors=['#087d86','#ae4774','#4966b2'];
  const source=key=>{const s=verifiedSource.excerpts[key];return `<details class="source-excerpt"><summary>核对源码 · ${s.path.split('/').at(-1)}:${s.start}</summary><p class="source-path">${s.path} · 本地快照，核对日期 ${verifiedSource.date}</p><pre><code>${s.text.split('\n').map((l,i)=>`${s.start+i}  ${esc(l)}`).join('\n')}</code></pre></details>`;};
  const hero=(n,title,desc)=>`<section class="hero lesson-hero"><div><div class="eyebrow">交互课程 ${n} / 06 · 源码与计算对照</div><h1>${title}</h1><p>${desc}</p></div><aside class="hero-aside"><strong>先建立整体，再逐项计算</strong><a href="#construction">05 模型如何搭建</a><br><a href="#network">06 隐藏层与细胞分布</a></aside></section>`;
  const formula=s=>`<div class="formula">${s}</div>`;
  let current='construction',step=0,network=new NetworkLab(),cell=0,unit=0,coord=0,axisX=0,axisY=1,serial=0,mode='centers',gene=0,delta=0,lambda=.1,ghost=null,runTimer=null,busy=false;
  let lastFit=null,latentHits=[],expressionHits=[];
  const constructionSteps=[
    {title:'登记观测',short:'S、U 与图先验',text:'每行是一个细胞，每列是一个基因。S（spliced，已剪接 RNA）与 U（unspliced，未剪接 RNA）来自同一批细胞、同一基因顺序。setup_anndata 登记表达层；基因筛选、归一化或平滑由上游分析决定。代码将层标记为 is_count_data=False，不能把任意原始计数直接当作已处理输入。先验调控图 W 则提供基因之间的候选连接信息。',formula:'S, U ∈ ℝ<sup>B × G</sup>，X = [S ; U] ∈ ℝ<sup>B × 2G</sup>',key:'registration'},
    {title:'共享编码',short:'2G → 256',text:'Encoder（编码器）用同一组权重处理每个细胞。默认隐藏层依次执行 Linear（线性变换）、BatchNorm（批归一化）、LayerNorm（层归一化）、ReLU（整流线性函数）和 Dropout（随机失活，概率 0.1）。前两种归一化稳定中间数值；ReLU 将负值置零；随机失活只在训练时启用。log_variational 默认 False，需要主动启用才做 log(0.01+x)。',formula:'a<sub>i</sub> = W<sub>h</sub>x<sub>i</sub> + b<sub>h</sub>，h<sub>i</sub> = Dropout(ReLU(LN(BN(a<sub>i</sub>))))',key:'layers'},
    {title:'定义后验',short:'μ、v → z',text:'Posterior（后验）描述看到表达后，这个细胞可能对应哪些潜变量。变分编码器用对角高斯 q 近似它：两个线性输出头各给出 D 个数，分别构成均值 μ 与方差 v。Softplus（平滑正值函数）把方差头输出变为正数。Reparameterization（重参数化）把随机性放在 ε 中，使 z 对网络参数可求导。',formula:'q<sub>φ</sub>(z<sub>i</sub> | x<sub>i</sub>) = 𝒩(μ<sub>i</sub>, diag(v<sub>i</sub>))<br>z<sub>i</sub> = μ<sub>i</sub> + √v<sub>i</sub> ⊙ ε<sub>i</sub>，ε<sub>i</sub> ∼ 𝒩(0, I)',key:'heads'},
    {title:'生成表达',short:'z → t → Ŝ、Û',text:'Decoder（解码器）先把 z 映射成各目标基因的时间比例 ρ，经过 Sigmoid（压到 0 与 1 之间的函数）后乘 t_max。GRN（基因调控网络）用当前已剪接表达计算转录率 α(s)。ODE（常微分方程）积分转录、剪接和降解过程，再在预测时间读取表达。z、逐基因时间 t 和 RNA 表达具有不同的维度与含义。',formula:'t<sub>ig</sub> = t<sub>max</sub> ρ<sub>ig</sub><br>du<sub>g</sub>/dt = α<sub>g</sub>(s) − β<sub>g</sub>u<sub>g</sub>，ds<sub>g</sub>/dt = β<sub>g</sub>u<sub>g</sub> − γ<sub>g</sub>s<sub>g</sub>',key:'time'},
    {title:'比较观测',short:'重建与约束',text:'Likelihood（似然）评价当前模型对观测表达的解释能力。源码用正态观测分布计算目标基因 S/U 的负对数似然，再加 KL divergence（KL 散度，对后验与标准正态先验差异的度量）。完整目标还包含速度约束、转录相关性、调控雅可比 L1 项、图先验和偏置项；各项由开关与权重决定，lam2 默认 0。',formula:'ℒ = mean(ℒ<sub>rec</sub> + λ<sub>KL</sub> KL + ℒ<sub>velo</sub>) + ℒ<sub>α</sub> + ℒ<sub>L1</sub> + ℒ<sub>graph</sub> + ℒ<sub>bias</sub>',key:'loss'},
    {title:'更新与验证',short:'梯度 → 新参数',text:'Backpropagation（反向传播）沿计算图用链式法则求梯度，参数更新后再重新计算每个细胞的表示。源码默认使用 AdamW（带权重衰减的自适应优化器）、90% 训练划分及验证集早停。Early stopping（早停）在验证指标长期不再改善时停止训练；训练损失下降仍需要验证数据和生物学检验。',formula:'∂ℒ<sub>rec</sub>/∂φ = (∂ℒ<sub>rec</sub>/∂x̂)(∂x̂/∂t)(∂t/∂z)(∂z/∂φ)<br>φ、时间解码器、调控权重及动力学参数共同参与拟合。',key:'training'}
  ];
  function constructionHTML(){return hero('05','先把 RegVelo 的整条链搭起来','同一份表达同时用于推断细胞状态和评价表达预测。下面沿源码走一轮，分清编码器、时间解码器与基因调控网络各自负责什么。')+`
    <div class="lesson-start"><b>建议阅读顺序</b><span>点下面六步看结构 → 调整张量尺寸 → 看梯度回传 → <a href="#network">进入隐藏层实验</a></span></div>
    <div class="flow six construction-flow" aria-label="RegVelo 构建与拟合六步">${constructionSteps.map((s,i)=>`<button data-build-step="${i}">${i+1}. ${s.title}<span>${s.short}</span></button>`).join('')}</div>
    <section class="panel" aria-live="polite"><div class="panel-head"><h2 id="build-title"></h2><span class="tag">真实源码结构</span></div><p id="build-text"></p><div id="build-formula"></div><div id="build-source"></div><div class="button-row mt"><button id="build-next" class="primary">看下一步</button><a href="#network">带着这条链去算一个细胞 →</a></div></section>
    <section class="panel wide-panel"><div class="panel-head"><div><h2>高维输入怎样变成低维张量？</h2><p class="sub">B 是一批细胞数，G 是保留的输入基因数。这里计算实际形状，不创建或训练大网络。细胞数 B 沿前向过程保持不变。</p></div><span class="tag gray">默认 H = 256，D = 10</span></div>
      <div class="grid3"><label class="control">细胞数 B<select id="shape-b"><option>32</option><option selected>128</option><option>512</option></select></label><label class="control">基因数 G<select id="shape-g"><option>500</option><option selected>2000</option><option>5000</option></select></label><label class="control">潜变量维度 D<select id="shape-d"><option>2</option><option selected>10</option><option>20</option></select></label></div>
      <div class="shape-chain" id="real-shapes"></div><div class="table-scroll"><table id="real-weights"></table></div><p class="tensor-note">W<sub>h</sub> 的每一行组合所有输入基因，产生一个隐藏特征；每个输出头再组合全部隐藏特征。输入被混合成新的坐标，单个潜维度没有天然的基因或通路名称。归一化参数另计。</p>${source('defaults')}${source('encoder')}${source('input')}
    </section>
    <section class="panel wide-panel"><h2>三个网络模块的参数，各自表达什么？</h2><div class="grid3"><div class="definition"><strong>编码器 z_encoder</strong><span>输入一行表达，输出这个细胞的 μ 与 v。连接权重表示特征组合，不能逐边直接解读成调控关系。</span></div><div class="definition"><strong>时间解码器 decoder</strong><span>输入 z，输出逐目标基因的时间比例 ρ。动力学据此在相应时间读取预测表达。</span></div><div class="definition"><strong>调控模块 v_encoder</strong><span>虽然命名包含 encoder，它负责计算转录率与 RNA 速度。先验图、调控权重和调控雅可比约束在这里发挥作用。</span></div></div>${source('ode')}</section>
    <section class="panel wide-panel"><div class="panel-head"><div><h2>一个拟合误差，怎样传回编码器？</h2><p class="sub">这块连接第一页的两基因动力学模型。点击按钮完成一次真实梯度更新，并查看各组参数的改动。此处目标为重建项 + 0.05 × KL。</p></div><span class="tag amber">两基因教学计算</span></div><div class="gradient-chain">表达误差 ← 动力学积分 ← 逐基因时间 ← z ← μ、v ← 编码器权重</div><div class="button-row mt"><button id="build-fit" class="primary">运行一轮拟合并追踪梯度</button><a href="#training">在第一页连续训练</a></div><p id="build-fit-status" class="status" role="status"></p><div class="table-scroll"><table id="build-gradients"></table></div><p class="tensor-note">数值是这份教学模型的导数与更新，不是手工指定的移动路径。共享权重更新会同时改变多个细胞的位置；单步目标可能上升。</p></section>
    <section class="recap"><h2>潜空间中的细胞分布从何而来？</h2><p>不同细胞有不同的表达向量。共享编码器把这些差异转成不同的分布中心与方差。重建表达的需求和先验约束共同塑造这些分布；模型没有接收“这个细胞应当画在这里”的坐标标签。</p></section>
    <div class="page-links"><a href="#encoder">← 04 线性编码算术</a><a href="#network">06 隐藏层与分布实验 →</a></div>`;}
  function refreshConstruction(){
    const s=constructionSteps[step];$('build-title').textContent=`${step+1}. ${s.title}`;$('build-text').textContent=s.text;$('build-formula').innerHTML=formula(s.formula);$('build-source').innerHTML=source(s.key);
    document.querySelectorAll('[data-build-step]').forEach(b=>{b.classList.toggle('active',+b.dataset.buildStep===step);b.setAttribute('aria-pressed',String(+b.dataset.buildStep===step));});
    const B=+$('shape-b').value,G=+$('shape-g').value,D=+$('shape-d').value;
    $('real-shapes').innerHTML=[`S、U 各 ${B} × ${G}`,`拼接 ${B} × ${2*G}`,`隐藏 ${B} × 256`,`μ、v 各 ${B} × ${D}`,`一次 Z ${B} × ${D}`].map(x=>`<b>${x}</b>`).join('<span>→</span>');
    $('real-weights').innerHTML=`<thead><tr><th>线性层</th><th>权重形状（输出 × 输入）</th><th>权重 + 偏置数量</th></tr></thead><tbody><tr><td>共享隐藏层</td><td>256 × ${2*G}</td><td>${(256*(2*G+1)).toLocaleString()}</td></tr><tr><td>均值头</td><td>${D} × 256</td><td>${(D*257).toLocaleString()}</td></tr><tr><td>方差头</td><td>${D} × 256</td><td>${(D*257).toLocaleString()}</td></tr></tbody>`;
    if(!lastFit){$('build-fit-status').textContent=`共享模型当前已更新 ${sharedModel.epoch} 次。点击查看下一轮。`;$('build-gradients').innerHTML='';}
    else{$('build-fit-status').textContent=lastFit.status;$('build-gradients').innerHTML=lastFit.table;}
  }
  function setupConstruction(){
    current='construction';document.querySelectorAll('[data-build-step]').forEach(b=>b.onclick=()=>{step=+b.dataset.buildStep;refreshConstruction();});
    $('build-next').onclick=()=>{step=(step+1)%6;refreshConstruction();};['shape-b','shape-g','shape-d'].forEach(id=>$(id).onchange=refreshConstruction);
    $('build-fit').onclick=()=>{
      const before=sharedModel.encoder(0).mu,packet=sharedModel.step({lambda:.05,fixedNoise:true}),after=sharedModel.evaluate({lambda:.05,fixedNoise:true});
      if(packet.error){$('build-fit-status').textContent=packet.error;return;}
      const rows=[['编码器均值权重','m00'],['编码器方差权重','v00'],['时间解码器权重','t00'],['调控权重','w00'],['剪接速率的原参数','beta0'],['降解速率的原参数','gamma0']];
      lastFit={status:`第 ${sharedModel.epoch} 次更新：同一噪声下总目标 ${f(packet.loss)} → ${f(after.loss)}。细胞 1 的 μ：${vector(before)} → ${vector(sharedModel.encoder(0).mu)}。`,table:`<thead><tr><th>参数组</th><th>更新前</th><th>∂ℒ/∂参数</th><th>更新后</th></tr></thead><tbody>${rows.map(([name,key])=>{const c=packet.changes[key];return `<tr><td>${name}</td><td>${f(c.before,5)}</td><td>${f(c.gradient,5)}</td><td>${f(c.after,5)}</td></tr>`;}).join('')}</tbody>`};refreshConstruction();
    };refreshConstruction();
  }
  const labels=()=>Array.from({length:24},(_,j)=>(j<12?'S':'U')+(j%12+1));
  function networkHTML(){return hero('06','让隐藏层的每一次计算可见','从 24 个表达特征开始，经过 8 个 ReLU 隐藏单元，计算每个细胞的均值与方差。选中一个单元看加权求和，再观察整批细胞怎样形成潜空间分布。')+`
    <div class="callout teal"><b>可训练的结构教学实验</b> · 36 个合成细胞，12 个合成基因。这里用小型 VAE（变分自编码器：用分布编码，再重建观测）直接重建表达，以便实时展开隐藏层。它省略 RegVelo 的时间解码、RNA 动力学、归一化、随机失活和图约束；完整链见<a href="#construction">第五页</a>。</div>
    <div class="toolbar mt"><div class="button-row"><label class="sub">细胞 <select id="net-cell">${network.data.map((c,i)=>`<option value="${i}">${c.name}</option>`).join('')}</select></label><label class="sub">潜维度 <select id="net-dim"><option value="2">2 维 · 全部可画</option><option value="10">10 维 · 同源码默认</option></select></label><label class="check"><input id="net-log" type="checkbox">log(0.01+x) 输入变换</label><button id="net-reset">重置此实验</button></div><span class="tag amber" id="net-state"></span></div>
    <p class="sub">改变潜维度或对数开关会重新初始化本实验。默认对数关闭，与核对的源码一致。合成细胞颜色只标识造数分组，不参与损失。</p>
    <section class="panel wide-panel"><div class="panel-head"><div><h2>一行表达，共享网络，两组输出</h2><p class="sub">选择中间的隐藏单元，查看哪些输入正向或负向贡献给它。每个数值都由当前参数实时计算。</p></div><span class="tag" id="net-shape"></span></div>
      <div class="neural-flow"><div><h3>① 输入 x · 24 维</h3><p class="sub">12 个 S + 12 个 U</p><div id="net-inputs" class="input-tiles"></div></div><div class="neural-arrow" aria-hidden="true">→</div><div><h3>② 隐藏 h · 8 维</h3><p class="sub">h = ReLU(Wx+b)</p><div id="net-hidden" class="hidden-tiles"></div></div><div class="neural-arrow" aria-hidden="true">→</div><div><h3>③ 分布参数</h3><p class="sub">点击一维，展开其输出头</p><div id="net-heads" class="head-tiles"></div></div></div>
      <div class="neuron-explain" id="net-neuron" aria-live="polite"></div><details><summary>展开这个隐藏单元的全部 24 项乘法</summary><div class="table-scroll"><table id="net-products"></table></div></details><div id="net-head-math" class="readout mt"></div><p class="tensor-note">这里的隐藏单元是学习到的特征组合。RegVelo 默认用 256 个单元，且在线性层后增加批归一化、层归一化，再做 ReLU 和训练期随机失活。</p>
    </section>
    <section class="panel wide-panel"><div class="panel-head"><div><h2>同一个细胞，表达位置与潜位置</h2><p class="sub">点任意细胞可同时定位两侧。表达图仅取 S1、U1 两轴，其余 22 个输入仍参与网络计算。潜空间的坐标对可以切换。</p></div></div>
      <div class="toolbar"><div class="button-row"><label class="sub">横轴 <select id="net-axis-x"></select></label><label class="sub">纵轴 <select id="net-axis-y"></select></label><label class="sub">潜空间显示 <select id="net-mode"><option value="centers">每个细胞的均值中心</option><option value="cell">选中细胞的后验样本</option><option value="aggregate">所有细胞的聚合后验</option></select></label><button id="net-sample">换一份采样噪声</button></div></div>
      <div class="grid2"><div><h3>观测表达 · 二维坐标投影</h3><canvas id="net-expression" class="chart tall" role="img" aria-label="36 个合成细胞在 S1 与 U1 上的表达投影"></canvas></div><div><h3>潜空间 · 所选两维的边缘分布</h3><canvas id="net-latent" class="chart tall" role="img" aria-label="细胞潜均值、所选细胞分布与聚合后验样本"></canvas></div></div>
      <div class="legend"><span><i></i>实心点：细胞均值 μ</span><span><i class="pink"></i>小点：后验样本</span><span><i class="amber"></i>橙色圈：选中细胞</span><span><i class="ghost"></i>灰色圈：上次训练前</span></div><div id="net-points-note" class="readout mt"></div>
      <div class="grid3 mt"><div class="definition"><strong>一个细胞的分布</strong><span>q<sub>i</sub> = 𝒩(μ<sub>i</sub>, diag(v<sub>i</sub>))。它描述给定这份表达时，潜变量可能的取值。</span></div><div class="definition"><strong>一批细胞的均值点</strong><span>μ<sub>1</sub>, …, μ<sub>N</sub> 是 N 个中心。中心可以沿曲线或多簇排列；这些点没有显示单个细胞的不确定性。</span></div><div class="definition"><strong>聚合后验</strong><span>Aggregated posterior（聚合后验）q<sub>agg</sub>(z) = (1/N) Σ<sub>i</sub>q<sub>i</sub>(z)。这里为每个细胞抽取相同数量的样本，可形成多峰分布。</span></div></div><p class="tensor-note">对角高斯只约束每个细胞内部的条件协方差；细胞中心之间仍可相关。图中椭圆的标准化半径为 1，在二维高斯下包围约 39.3% 的概率，并非 68%。这些图直接显示潜坐标，没有另做 UMAP（二维流形可视化算法）。</p>
    </section>
    <section class="panel wide-panel"><h2>改动一个输入，追踪位置响应</h2><p>选中细胞的一项表达变化后，同一组权重会立即重新计算 h、μ 与 v。这是编码器敏感性演示，不能据此解释成基因敲除的因果效应。</p><div class="grid2"><label class="control">选择表达特征<select id="net-gene">${labels().map((l,i)=>`<option value="${i}">${l}</option>`).join('')}</select></label><label class="control"><span>表达增量 Δ（下限截到 0.001）<output id="net-delta-value">0.00</output></span><input id="net-delta" type="range" min="-1" max="1" value="0" step=".02"></label></div><button id="net-restore">恢复观测输入</button><p id="net-perturb" class="status" role="status"></p><p class="sub">扰动仅作用于当前前向预览。训练使用原始 36 行表达；点击训练会清除预览扰动。</p></section>
    <section class="panel wide-panel"><div class="panel-head"><div><h2>拟合数据后，细胞为什么会重新排列？</h2><p class="sub">沿“表达 → 隐藏层 → z → 重建表达”的计算图真实求导。固定采样噪声用于比较更新前后；训练不使用颜色分组或目标潜坐标。</p></div><span class="tag amber">本页独立模型</span></div>
      <div class="toolbar"><div class="button-row"><button id="net-train-one" class="primary">更新 1 步</button><button id="net-train">更新 80 步</button><button id="net-stop">暂停</button></div><label class="sub">KL 权重 <select id="net-lambda"><option value="0">0</option><option value=".1" selected>0.1</option><option value="1">1</option><option value="5">5</option></select></label></div>
      <div class="metrics"><div class="metric"><span class="label">已更新</span><strong id="net-epoch"></strong></div><div class="metric"><span class="label">当前总目标</span><strong id="net-loss"></strong></div><div class="metric"><span class="label">当前重建项</span><strong id="net-rec"></strong></div><div class="metric"><span class="label">当前 KL</span><strong id="net-kl"></strong></div></div><div class="grid2"><div><canvas id="net-loss-chart" class="chart" role="img" aria-label="本页合成数据模型的总目标与重建项随更新变化"></canvas><div class="legend"><span><i></i>总目标</span><span><i class="amber"></i>重建项</span></div></div><div><h3>当前细胞 · 前 3 个基因的重建</h3><div class="table-scroll"><table id="net-reconstruction"></table></div><p class="sub">全部 24 项都参与损失，表中仅显示前 3 个基因的 S/U。观测分布为标准差 0.3 的正态分布；解码器用 Softplus 生成正均值。</p></div></div>
      <p id="net-train-status" class="status" role="status"></p>${formula('ℒ = (1/N) Σ<sub>i</sub>[−log p<sub>θ</sub>(x<sub>i</sub> | z<sub>i</sub>) + λ KL(q<sub>φ</sub>(z<sub>i</sub> | x<sub>i</sub>) ‖ 𝒩(0,I))]')}<p>重建项要求 z 保留有助于解释表达的信息；KL 项牵引均值与方差接近先验。调大 λ 不保证得到更清晰的分群。训练只优化当前目标，潜维度的生物意义需要后续检验。</p>
    </section>
    <details><summary>展开整批 μ 与 v 的数值矩阵</summary><p>一行一个细胞。切换 10 维时，每个细胞有 10 个均值与 10 个方差。选中行使用当前预览输入。</p><div class="table-scroll"><table id="net-matrix"></table></div></details>
    <details><summary>复习：训练时的随机性，与提取结果时的确定性</summary><p>真实 RegVelo 训练时启用 Dropout，BatchNorm 使用批次统计，z 还要采样。用评估模式提取 μ 时，Dropout 关闭，BatchNorm 使用保存的统计；固定表达与参数可得到稳定的均值。重复抽 z 仍然会变化。</p><p>本页没有归一化与随机失活，固定表达和权重时 μ、v 始终固定。“换一份采样噪声”仅改变可视化样本与所展示的重建；训练与损失曲线始终使用固定的另一份噪声。</p></details>
    <div class="page-links"><a href="#construction">← 05 RegVelo 的完整拟合链</a><a href="#space">02 潜变量在程序里的张量形式 →</a></div>`;}
  function stop(){if(runTimer!==null)clearTimeout(runTimer);runTimer=null;busy=false;}
  function resetNetwork(){stop();network=new NetworkLab(+$('net-dim').value,$('net-log').checked);axisX=0;axisY=1;coord=0;serial=0;delta=0;ghost=null;$('net-delta').value=0;axisOptions();refreshNetwork();$('net-train-status').textContent='已重新初始化本页网络。';}
  function axisOptions(){for(const id of ['net-axis-x','net-axis-y'])$(id).innerHTML=Array.from({length:network.dim},(_,i)=>`<option value="${i}">第 ${i+1} 维</option>`).join('');$('net-axis-x').value=axisX;$('net-axis-y').value=axisY;}
  function traces(){return network.data.map((_,i)=>network.trace(i,{serial,...(i===cell?{gene,delta}:{})}));}
  function tracePanels(rows){
    const t=rows[cell],names=labels(),p=network.p;
    $('net-inputs').innerHTML=t.x.map((v,j)=>{const contribution=v*p[`h${unit}_${j}`];return `<span class="input-tile ${contribution>=0?'positive':'negative'}" title="${names[j]} 对 h${unit+1} 的贡献 ${f(contribution,5)}"><b>${names[j]}</b> ${f(v,2)}</span>`;}).join('');
    $('net-hidden').innerHTML=t.h.map((v,j)=>`<button data-unit="${j}" class="${j===unit?'active':''}" aria-pressed="${j===unit}"><span>h${j+1}</span><b>${f(v)}</b><small>${t.a[j]<=0?'ReLU 置零':'保留正值'}</small></button>`).join('');
    document.querySelectorAll('[data-unit]').forEach(b=>b.onclick=()=>{unit=+b.dataset.unit;refreshNetwork(false);});
    $('net-heads').innerHTML=t.mu.map((v,j)=>`<button data-coordinate="${j}" class="${coord===j?'active':''}" aria-pressed="${coord===j}"><b>维 ${j+1}</b><span>μ ${f(v)} · v ${f(t.variance[j])}</span></button>`).join('');
    document.querySelectorAll('[data-coordinate]').forEach(b=>b.onclick=()=>{coord=+b.dataset.coordinate;refreshNetwork(false);});
    const sum=t.x.reduce((s,v,j)=>s+v*p[`h${unit}_${j}`],0),terms=t.x.map((v,j)=>({label:names[j],value:v*p[`h${unit}_${j}`]})).sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)).slice(0,3);
    $('net-neuron').innerHTML=`<h3>隐藏单元 h${unit+1} 的一整次计算</h3>${formula(`a<sub>${unit+1}</sub> = Σ<sub>j=1…24</sub> W<sub>${unit+1},j</sub>x<sub>j</sub> + b<sub>${unit+1}</sub> = ${f(sum)} + (${f(p['hb'+unit])}) = ${f(t.a[unit])}<br>h<sub>${unit+1}</sub> = max(0, ${f(t.a[unit])}) = ${f(t.h[unit])}`)}<p class="sub">绝对贡献最大的三项：${terms.map(x=>`${x.label}：${f(x.value)}`).join('；')}。绿色输入块为非负贡献，粉色为负贡献。ReLU 的门由当前整行输入共同决定。</p>`;
    $('net-products').innerHTML=`<thead><tr><th>特征</th><th>x</th><th>W</th><th>W × x</th></tr></thead><tbody>${t.x.map((v,j)=>`<tr><td>${names[j]}</td><td>${f(v,5)}</td><td>${f(p[`h${unit}_${j}`],5)}</td><td>${f(v*p[`h${unit}_${j}`],5)}</td></tr>`).join('')}</tbody>`;
    const expand=tag=>t.h.map((v,j)=>`(${f(p[`${tag}${coord}_${j}`])} × ${f(v)})`).join(' + ');
    $('net-head-math').innerHTML=`<b>第 ${coord+1} 维 · 两个输出头共享同一条 h</b><br>μ = ${f(p['mub'+coord])} + ${expand('mu')} = <b>${f(t.mu[coord])}</b><br>方差头原值 aᵥ = ${f(p['vab'+coord])} + ${expand('va')} = ${f(t.vraw[coord])}<br>v = log(1 + exp(${f(t.vraw[coord])})) + 0.0001 = <b>${f(t.variance[coord])}</b><br>z = ${f(t.mu[coord])} + √${f(t.variance[coord])} × (${f(t.eps[coord])}) = <b>${f(t.z[coord])}</b><br><span class="sub">显示值四舍五入，计算使用完整精度；μ、v 各有 ${network.dim} 项。</span>`;
  }
  function chart(id,xvals,yvals,xlabel,ylabel){
    const c=$(id),w=c.clientWidth,h=c.clientHeight,d=devicePixelRatio||1;c.width=w*d;c.height=h*d;const g=c.getContext('2d');g.scale(d,d);
    let xmin=Math.min(...xvals),xmax=Math.max(...xvals),ymin=Math.min(...yvals),ymax=Math.max(...yvals);
    const dx=Math.max(.1,xmax-xmin)*1.18,dy=Math.max(.1,ymax-ymin)*1.18,cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;
    const l=49,r=18,top=24,bottom=36,scale=Math.min((w-l-r)/dx,(h-top-bottom)/dy);
    xmin=cx-(w-l-r)/scale/2;xmax=cx+(w-l-r)/scale/2;ymin=cy-(h-top-bottom)/scale/2;ymax=cy+(h-top-bottom)/scale/2;
    const X=x=>l+(x-xmin)*scale,Y=y=>h-bottom-(y-ymin)*scale;
    g.font='10px "Microsoft YaHei", sans-serif';g.fillStyle='#607080';g.strokeStyle='#e2e8ed';
    for(let i=0;i<5;i++){const x=xmin+(xmax-xmin)*i/4,y=ymin+(ymax-ymin)*i/4;g.beginPath();g.moveTo(X(x),top);g.lineTo(X(x),h-bottom);g.moveTo(l,Y(y));g.lineTo(w-r,Y(y));g.stroke();g.textAlign='center';g.fillText(f(x,1),X(x),h-bottom+14);g.textAlign='right';g.fillText(f(y,1),l-5,Y(y)+3);}
    g.textAlign='center';g.fillText(xlabel,w/2,h-4);g.textAlign='left';g.fillText(ylabel,8,14);
    return {g,X,Y,w,h,scale};
  }
  function point(p,x,y,color,r=4,hollow=false){p.g.beginPath();p.g.arc(p.X(x),p.Y(y),r,0,Math.PI*2);p.g.fillStyle=color;p.g.strokeStyle=color;p.g.lineWidth=1.7;hollow?p.g.stroke():p.g.fill();}
  function drawSpaces(rows){
    const observed=network.data.map(c=>[c.raw[0],c.raw[12]]),t=rows[cell],base=network.trace(cell);
    const left=chart('net-expression',[...observed.map(x=>x[0]),t.raw[0]],[...observed.map(x=>x[1]),t.raw[12]],'S1 · 已剪接表达','U1 · 未剪接表达');
    expressionHits=observed.map((p,i)=>{point(left,...p,colors[network.data[i].group]);return {i,x:left.X(p[0]),y:left.Y(p[1])};});point(left,t.raw[0],t.raw[12],'#ae6b13',8,true);
    const samples=[],r=new RNG(357+serial*911);
    if(mode!=='centers'){rows.forEach((row,i)=>{if(mode==='cell'&&i!==cell)return;for(let n=0;n<(mode==='cell'?180:12);n++)samples.push({i,x:row.mu[axisX]+Math.sqrt(row.variance[axisX])*r.normal(),y:row.mu[axisY]+Math.sqrt(row.variance[axisY])*r.normal()});});}
    const boundsX=rows.map(t=>t.mu[axisX]),boundsY=rows.map(t=>t.mu[axisY]);
    for(const k of [-1,1]){boundsX.push(t.mu[axisX]+k*Math.sqrt(t.variance[axisX]));boundsY.push(t.mu[axisY]+k*Math.sqrt(t.variance[axisY]));}
    boundsX.push(base.mu[axisX],t.z[axisX],...samples.map(s=>s.x));boundsY.push(base.mu[axisY],t.z[axisY],...samples.map(s=>s.y));
    if(ghost){boundsX.push(...ghost.map(v=>v[axisX]));boundsY.push(...ghost.map(v=>v[axisY]));}
    const right=chart('net-latent',boundsX,boundsY,`潜坐标 ${axisX+1}`,`潜坐标 ${axisY+1}`);
    if(ghost)ghost.forEach(p=>point(right,p[axisX],p[axisY],'#aebbc5',3,true));
    right.g.globalAlpha=.35;samples.forEach(s=>point(right,s.x,s.y,mode==='cell'?'#ae4774':colors[network.data[s.i].group],2));right.g.globalAlpha=1;
    latentHits=rows.map((t,i)=>{point(right,t.mu[axisX],t.mu[axisY],colors[network.data[i].group]);return {i,x:right.X(t.mu[axisX]),y:right.Y(t.mu[axisY])};});
    const g=right.g;g.strokeStyle='#ae6b13';g.lineWidth=1.5;g.beginPath();g.ellipse(right.X(t.mu[axisX]),right.Y(t.mu[axisY]),right.scale*Math.sqrt(t.variance[axisX]),right.scale*Math.sqrt(t.variance[axisY]),0,0,Math.PI*2);g.stroke();point(right,t.mu[axisX],t.mu[axisY],'#ae6b13',8,true);point(right,t.z[axisX],t.z[axisY],'#ae4774',5);
    if(delta)point(right,base.mu[axisX],base.mu[axisY],'#33485a',6,true);
    $('net-points-note').textContent=`${network.data[cell].name}：μ = ${vector(t.mu)}；v = ${vector(t.variance)}。粉色大点是当前一次样本 z。${mode==='aggregate'?'当前为每个细胞各抽 12 次的等权混合，共 432 个小点。':mode==='cell'?'当前 180 个小点全部来自同一个细胞的 q。':'当前显示 36 个细胞均值。'}${network.dim>2?` 本图只取第 ${axisX+1}、${axisY+1} 维，其余 8 维仍参与重建。`:''}`;
  }
  function drawLoss(){
    const c=$('net-loss-chart'),w=c.clientWidth,h=c.clientHeight,d=devicePixelRatio||1;c.width=w*d;c.height=h*d;const g=c.getContext('2d');g.scale(d,d);
    const hist=network.history;g.font='11px "Microsoft YaHei",sans-serif';g.fillStyle='#607080';if(!hist.length){g.fillText('点击“更新 80 步”，观察实时计算的目标。',16,30);return;}
    const vals=hist.flatMap(r=>[r.loss,r.reconstruction]),lo=Math.min(...vals)-1,hi=Math.max(...vals)+1,X=n=>47+(w-67)*(n-hist[0].epoch)/Math.max(1,hist.at(-1).epoch-hist[0].epoch),Y=v=>h-34-(h-62)*(v-lo)/(hi-lo);
    g.strokeStyle='#e2e8ed';for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;g.beginPath();g.moveTo(47,Y(v));g.lineTo(w-20,Y(v));g.stroke();g.fillText(f(v,1),2,Y(v)+4);}
    for(const [key,color] of [['loss','#087d86'],['reconstruction','#ae6b13']]){g.strokeStyle=color;g.beginPath();hist.forEach((v,i)=>i?g.lineTo(X(v.epoch),Y(v[key])):g.moveTo(X(v.epoch),Y(v[key])));g.stroke();if(hist.length===1){g.fillStyle=color;g.beginPath();g.arc(X(hist[0].epoch),Y(hist[0][key]),3,0,7);g.fill();}}
    g.fillStyle='#607080';g.fillText(`${hist[0].epoch}`,47,h-16);g.textAlign='right';g.fillText(`${hist.at(-1).epoch} 次更新前`,w-20,h-16);
  }
  let metrics=null;
  function refreshNetwork(recompute=true){
    const rows=traces(),t=rows[cell];if(recompute||!metrics)metrics=network.evaluate({lambda});
    $('net-cell').value=cell;$('net-dim').value=network.dim;$('net-log').checked=network.logInput;$('net-mode').value=mode;$('net-lambda').value=lambda;
    $('net-shape').textContent=`24 → 8 → ${network.dim} + ${network.dim}`;$('net-state').textContent=`36 个细胞 · 更新 ${network.epoch} 次`;
    $('net-delta-value').textContent=f(delta,2);const base=network.trace(cell);
    $('net-perturb').textContent=`${labels()[gene]}：原观测 ${f(base.raw[gene])} → 当前预览 ${f(t.raw[gene])}。μ 的移动距离 ${f(Math.hypot(...t.mu.map((v,j)=>v-base.mu[j])),5)}（全部 ${network.dim} 维）。`;
    tracePanels(rows);drawSpaces(rows);drawLoss();
    $('net-epoch').textContent=network.epoch;for(const [id,key] of [['net-loss','loss'],['net-rec','reconstruction'],['net-kl','kl']])$(id).textContent=f(metrics[key],2);
    $('net-reconstruction').innerHTML=`<thead><tr><th>特征</th><th>观测</th><th>当前样本重建</th></tr></thead><tbody>${[0,1,2,12,13,14].map(j=>`<tr><td>${labels()[j]}</td><td>${f(t.raw[j])}</td><td>${f(t.prediction[j])}</td></tr>`).join('')}</tbody>`;
    $('net-matrix').innerHTML=`<thead><tr><th>细胞</th>${Array.from({length:network.dim},(_,j)=>`<th>μ${j+1}</th><th>v${j+1}</th>`).join('')}</tr></thead><tbody>${rows.map((t,i)=>`<tr class="${i===cell?'selected':''}"><td>${i+1}</td>${t.mu.map((v,j)=>`<td>${f(v)}</td><td>${f(t.variance[j])}</td>`).join('')}</tr>`).join('')}</tbody>`;
    ['net-train-one','net-train','net-dim','net-log','net-lambda'].forEach(id=>$(id).disabled=busy);$('net-stop').disabled=!busy;
  }
  function train(n){
    stop();delta=0;$('net-delta').value=0;ghost=network.data.map((_,i)=>network.trace(i).mu);const before=network.evaluate({lambda}).loss;busy=true;let remaining=n;
    const tick=()=>{if(current!=='network'){stop();return;}try{const count=Math.min(4,remaining);for(let k=0;k<count;k++)network.step({lambda});remaining-=count;if(!remaining){busy=false;runTimer=null;}refreshNetwork();$('net-train-status').textContent=`本轮已更新 ${n-remaining}/${n} 步。相同噪声下目标 ${f(before)} → ${f(metrics.loss)}；灰色空心点保存本轮训练前的位置。${remaining?'可随时暂停。':'更新完成。'}`;if(remaining)runTimer=setTimeout(tick,15);}catch(e){stop();refreshNetwork();$('net-train-status').textContent=e.message;}};runTimer=setTimeout(tick,0);
  }
  function setupNetwork(){
    current='network';axisOptions();$('net-gene').value=gene;$('net-delta').value=delta;
    $('net-cell').onchange=()=>{cell=+$('net-cell').value;delta=0;$('net-delta').value=0;refreshNetwork(false);};
    $('net-dim').onchange=resetNetwork;$('net-log').onchange=resetNetwork;$('net-reset').onclick=resetNetwork;
    for(const id of ['net-axis-x','net-axis-y'])$(id).onchange=()=>{axisX=+$('net-axis-x').value;axisY=+$('net-axis-y').value;if(axisX===axisY){if(id==='net-axis-x')axisY=(axisX+1)%network.dim;else axisX=(axisY+1)%network.dim;}axisOptions();refreshNetwork(false);};
    $('net-mode').onchange=()=>{mode=$('net-mode').value;refreshNetwork(false);};$('net-sample').onclick=()=>{serial++;refreshNetwork(false);};
    $('net-gene').onchange=()=>{gene=+$('net-gene').value;refreshNetwork(false);};$('net-delta').oninput=()=>{delta=+$('net-delta').value;refreshNetwork(false);};$('net-restore').onclick=()=>{delta=0;$('net-delta').value=0;refreshNetwork(false);};
    $('net-train-one').onclick=()=>train(1);$('net-train').onclick=()=>train(80);$('net-stop').onclick=()=>{stop();refreshNetwork(false);$('net-train-status').textContent=`已暂停在第 ${network.epoch} 步，参数保留。`;};
    $('net-lambda').onchange=()=>{lambda=+$('net-lambda').value;network.history=[];refreshNetwork();$('net-train-status').textContent='KL 权重已改变，清空旧目标曲线以免混淆；保留当前参数。';};
    for(const id of ['net-expression','net-latent'])$(id).onclick=e=>{const b=e.currentTarget.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top,hits=id==='net-expression'?expressionHits:latentHits;const nearest=hits.map(p=>({...p,d:(p.x-x)**2+(p.y-y)**2})).sort((a,b)=>a.d-b.d)[0];if(nearest.d<225){cell=nearest.i;delta=0;$('net-delta').value=0;refreshNetwork(false);}};
    refreshNetwork();$('net-train-status').textContent='模型状态在本次打开页面期间保留。切换页面会暂停训练。';
  }
  return {constructionHTML,networkHTML,setupConstruction,setupNetwork,refreshConstruction,refreshNetwork,stop:()=>{stop();current='';},state:()=>({network,cell,unit,coord,axisX,axisY,delta,serial,mode,busy})};
}
