import {createSpatialLab} from './spatial-lab.mjs';
import {createKineticsLab} from './kinetics-lab.mjs';
import {createReplicaLab} from './replica-lab.mjs';
import {createStabilityLab} from './stability-lab.mjs';
import {createGeometryLab} from './geometry-lab.mjs';
import {stabilityReplicas} from './stability-replicas.mjs';
import {createPaperLab} from './paper-lab.mjs';

const spatialLab=createSpatialLab();
const kineticsLab=createKineticsLab();
const replicaLab=createReplicaLab(stabilityReplicas,{title:'五组代表模型 · 同一细胞的三个空间'});
const representativeLab=createReplicaLab(stabilityReplicas,{title:'五组代表模型 · 同一细胞的三个空间'});
const stabilityLab=createStabilityLab();
const geometryLab=createGeometryLab();
const paperLab=createPaperLab();
const routes=['replicas','stability','geometry','paper','replica-space','kinetics','replica-source','replica-compare','replica-encoder','replica-training','replica-kinetics','overview','encoder','training','space','construction'];
const aliases={network:'encoder',posterior:'space'};
const titles={overview:'三空间总览',encoder:'编码与解码',training:'拟合过程',space:'三维潜空间与分布',kinetics:'两池动力学与潜时间',construction:'RegVelo 对应',stability:'20次稳定性实验',geometry:'凸包与椭球定义',paper:'论文实验：先验到网络',replicas:'五行三空间总览','replica-compare':'五组独立拟合 · 同一细胞的三个空间','replica-space':'编码、训练与重参数化'};
let active='replicas',activeLab=representativeLab,resizeTimer;

function renderRoute(){
  spatialLab.stop();
  kineticsLab.stop();
  replicaLab.stop();
  stabilityLab.stop();
  geometryLab.stop();
  paperLab.stop();
  const requested=location.hash.slice(1)||'replicas';
  active=aliases[requested]??(routes.includes(requested)?requested:'replicas');
  activeLab=active==='replicas'?representativeLab:active==='stability'?stabilityLab:active==='geometry'?geometryLab:active==='paper'?paperLab:active.startsWith('replica')?replicaLab:active==='kinetics'?kineticsLab:spatialLab;
  document.getElementById('main').innerHTML=activeLab.template(active);
  document.querySelectorAll('[data-route]').forEach(a=>a.setAttribute('aria-current',a.dataset.route===active?'page':'false'));
  document.title=(titles[active]??'三次独立拟合')+' · RegVelo 三空间实验室';
  activeLab.setup(active);
  window.__latentLab={route:active,state:activeLab.state,get model(){return activeLab.state().model;},navigate:route=>{location.hash=route;}};
}
window.addEventListener('hashchange',()=>{renderRoute();window.scrollTo(0,0);});
window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>activeLab.refresh(),100);});
renderRoute();

