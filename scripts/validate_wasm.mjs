// Usage: node scripts/validate_wasm.mjs /path/to/@mujoco/mujoco/mujoco.js
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

if (!process.argv[2]) throw new Error('Pass the installed MuJoCo WASM module path.');
const {default: load} = await import(pathToFileURL(path.resolve(process.argv[2])).href);
const mj = await load();
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const directory = path.join(root, 'output/athletics/mujoco');
const vfs = new mj.MjVFS();
for (const p of ['scene.xml', 'field.xml', 'assets/carpet.png']) {
  vfs.addBuffer(p, new Uint8Array(fs.readFileSync(path.join(directory, p))));
}
const model = mj.MjModel.from_xml_path('scene.xml', vfs);
if (model.ngeom !== 35 || model.nsite !== 3 || model.nq !== 0) throw new Error('Unexpected field structure.');
const report = {engine:'MuJoCo WASM', version:mj.mj_versionString(), ngeom:model.ngeom, nsite:model.nsite, nq:model.nq, load:'passed'};
fs.writeFileSync(path.join(root, 'output/athletics/validation_wasm.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
model.delete();
vfs.delete();
