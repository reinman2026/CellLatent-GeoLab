// Build 离线版.html from dist/: inline the stylesheet, inline core.mjs + app.mjs
// as one classic script (module imports removed), and point the footer link at
// dist/source-map.json. Run: node tools/build-offline.mjs
import {readFileSync,writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname,join} from 'node:path';

const root=join(dirname(fileURLToPath(import.meta.url)),'..');
const read=p=>readFileSync(join(root,p),'utf8');
const html=read('dist/index.html'),css=read('dist/style.css');
const core=read('dist/core.mjs'),app=read('dist/app.mjs');

const stripExports=src=>src.replace(/^export (class|function|const|let) /gm,'$1 ');
const appBody=app.replace(/^import[^\n]*\n/,'').replace(/^\n+/,'');

const out=html
  .replace('<link rel="stylesheet" href="style.css">',`<style>\n${css.trimEnd()}\n</style>`)
  .replace('<script type="module" src="app.mjs"></script>',`<script>\n${stripExports(core).trimEnd()}\n\n\n${appBody.trimEnd()}\n</script>`)
  .replace('href="source-map.json"','href="dist/source-map.json"');

if(/^export /m.test(stripExports(core))||/^import /m.test(appBody))throw new Error('offline bundle still contains a module statement');
writeFileSync(join(root,'离线版.html'),out,'utf8');
console.log(`离线版.html written: ${out.length} chars`);
