// Build 离线版.html from dist/: inline the stylesheet, source excerpts and all
// application modules as one classic script. Resolve companion footer files
// under dist/. Run: node tools/build-offline.mjs
import {readFileSync,writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname,join} from 'node:path';

const root=join(dirname(fileURLToPath(import.meta.url)),'..');
const read=p=>readFileSync(join(root,p),'utf8');
const html=read('dist/index.html'),css=read('dist/style.css');
const core=read('dist/core.mjs'),app=read('dist/app.mjs');
const sources=read('dist/verified-source.mjs'),spatial=read('dist/spatial-lab.mjs');
const kinetics=read('dist/kinetics-lab.mjs');
const paperToy=read('dist/paper-toy-data.mjs');
const geometry=read('dist/geometry3d.mjs');
const stability=read('dist/stability-lab.mjs');
const geometryLab=read('dist/geometry-lab.mjs');
const replicaUI=read('dist/replica-lab.mjs');
const stabilityReplicaData=read('dist/stability-replicas.mjs');
const paperToyStability=read('dist/paper-toy-stability-results.mjs');
const paperToyWHistory=read('dist/paper-toy-w-history.mjs');
const paperLab=read('dist/paper-lab.mjs');

const stripExports=src=>src.replace(/^export (class|function|const|let) /gm,'$1 ');
const stripImports=src=>src.replace(/^import[^\n]*\n/gm,'').replace(/^\n+/,'');
const appBody=stripImports(app);
const bundle=[core,sources,stripExports(paperToy),paperToyWHistory,paperToyStability,stripImports(spatial),stripImports(kinetics),stripExports(geometry),stripImports(stability),stripImports(geometryLab),stripImports(stabilityReplicaData),stripImports(replicaUI),stripImports(paperLab),appBody].map(stripExports).join('\n\n');

const out=html
  .replace('<link rel="stylesheet" href="style.css">',()=>`<style>\n${css.trimEnd()}\n</style>`)
  .replace('<script type="module" src="app.mjs"></script>',()=>`<script>\n${bundle.trimEnd()}\n</script>`)
  .replace('href="source-map.json"','href="dist/source-map.json"')
  .replace('href="模型路线图.html"','href="dist/模型路线图.html"');

if(/^(export|import) /m.test(bundle))throw new Error('offline bundle still contains a module statement');
writeFileSync(join(root,'离线版.html'),out,'utf8');
console.log(`离线版.html written: ${out.length} chars`);

