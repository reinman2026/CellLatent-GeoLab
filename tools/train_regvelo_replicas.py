"""Fit complete local RegVelo models; export exact posterior forward passes.

Run with .runtime/regvelo-venv/Scripts/python.exe after installing requirements.
The same observations, gene order, prior and train/validation split are used in
all runs. Only model/training random seeds differ. No teaching ODE is used here.
The observation JSON may provide a target-by-regulator ``prior.skeleton`` and
model options such as ``soft_constraint``, ``lam``, ``lam2`` and
``simple_dynamics``.  The legacy three-gene website JSON remains supported.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
import re
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('E:/PycharmProjects/rna_window_response_hlc_snapshot-regnet/ref/regvelo')
SEEDS = tuple(11 + 18 * index for index in range(100))
REQUIRED = ('torch', 'anndata', 'scvi', 'torchode', 'lightning', 'cellrank')
sys.dont_write_bytecode = True
# The supplied local venv needs this for scvi/scanpy imports. This disables
# Numba's optional JIT compiler, not PyTorch autograd or the RegVelo ODE.
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(source: Path, probe_imports: bool = False) -> dict:
    modules = {name: importlib.util.find_spec(name) is not None for name in REQUIRED}
    versions = {}
    for name in ('torch', 'scvi-tools', 'torchode', 'lightning', 'cellrank', 'numpy'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    incompatible = []
    for name, maximum in (('torch', (2, 6, 0)), ('scvi-tools', (1, 2, 1))):
        installed = versions[name]
        numbers = tuple(int(x) for x in re.findall(r'\d+', installed.split('+')[0])[:3]) if installed else ()
        if numbers and (numbers + (0, 0, 0))[:3] >= maximum:
            incompatible.append(f'{name}=={installed}; local RegVelo requires <' + '.'.join(map(str, maximum)))
    source_exists = (source / 'src/regvelo/_model.py').is_file()
    # Package constraints describe the source's supported versions. Newer
    # versions are reported, then tested through imports and actual training.
    ready = all(modules.values()) and source_exists
    import_error = None
    imports_checked = probe_imports and ready
    if imports_checked:
        try:
            for name in ('ctypes', 'numpy', *REQUIRED):
                importlib.import_module(name)
            sys.path.insert(0, str(source / 'src'))
            importlib.import_module('regvelo')
        except Exception:
            import_error = traceback.format_exc()
            ready = False
    return {'python': sys.executable, 'modules': modules, 'versions': versions,
            'versionWarnings': incompatible, 'declaredCompatible': not incompatible,
            'importsChecked': imports_checked, 'numbaDisableJit': os.environ.get('NUMBA_DISABLE_JIT'),
            'importError': import_error,
            'source': str(source), 'source_exists': (source / 'src/regvelo/_model.py').is_file(),
            'ready': ready}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_observations(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    genes = data.get('genes')
    feature_order = data.get('featureOrder')
    if not isinstance(genes, list) or not genes or len(set(genes)) != len(genes):
        raise ValueError('Expected a non-empty list of unique gene names.')
    expected_features = [f'S_{gene}' for gene in genes] + [f'U_{gene}' for gene in genes]
    if (not isinstance(feature_order, list) or len(feature_order) != 2 * len(genes)
            or len(set(feature_order)) != len(feature_order)
            or set(feature_order) != set(expected_features)):
        raise ValueError('featureOrder must contain exactly one S_* and one U_* feature for every gene.')
    n_genes = len(genes)
    if len(data.get('cells', [])) < 12:
        raise ValueError('At least 12 cells are required for shared train/validation subsets.')
    for cell in data['cells']:
        if len(cell['raw']) != 2 * n_genes or any(not isinstance(v, (int, float)) or not 0 <= v < 1e8 for v in cell['raw']):
            raise ValueError(f'Invalid {2 * n_genes}-dimensional expression observation.')
    if len({c['id'] for c in data['cells']}) != len(data['cells']):
        raise ValueError('Cell identifiers must be unique.')
    prior = data.get('prior') or {}
    orientation = prior.get('orientation')
    if orientation is not None and orientation not in ('target_by_regulator', 'target-by-regulator'):
        raise ValueError('prior.orientation must be target_by_regulator for the local VELOVAE weight layout.')
    skeleton = prior.get('skeleton')
    if skeleton is not None:
        if (not isinstance(skeleton, list) or len(skeleton) != n_genes
                or any(not isinstance(row, list) or len(row) != n_genes for row in skeleton)):
            raise ValueError('prior.skeleton must be a square target-by-regulator matrix.')
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for row in skeleton for value in row):
            raise ValueError('prior.skeleton contains a non-finite value.')
    return data


def resolve_model_settings(data: dict) -> dict:
    """Read model settings while accepting both current and toy JSON layouts."""
    prior = data.get('prior') or {}
    model = data.get('model') or data.get('training') or {}

    def pick(name, default):
        if name in model:
            return model[name]
        if name in data:
            return data[name]
        return prior.get(name, default)

    soft_constraint = bool(pick('soft_constraint', True))
    lam = float(pick('lam', 1.0))
    lam2 = float(pick('lam2', 0.0))
    simple_dynamics = bool(pick('simple_dynamics', False))
    regulators = pick('regulators', None)
    if regulators is not None and not isinstance(regulators, list):
        raise ValueError('regulators must be a list of gene names when provided.')
    return {'soft_constraint': soft_constraint, 'lam': lam, 'lam2': lam2,
            'simple_dynamics': simple_dynamics, 'regulators': regulators}


def fit_one(job: dict) -> dict:
    """Executed in an independent process with its own torch RNG and optimizer."""
    import numpy as np
    import pandas as pd
    import torch
    import anndata as ad
    import scvi
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from unittest.mock import patch
    from functools import partial
    from scvi.dataloaders import DataSplitter

    source = Path(job['source'])
    # scvi 1.5's checkpoint callback creates ``scvi_log`` relative to the
    # process cwd. Keep that generated metadata inside the run directory,
    # which is already granted low-integrity write access by the launcher.
    output = Path(job['output']) / f'model-{job["index"] + 1}'
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    # scvi's checkpoint callback calls ``model.save`` under this directory.
    # Explicitly redirect it because the v1.5 callback otherwise resolves the
    # default ``./scvi_log`` against the parent process cwd.
    # Keep this path short: scvi 1.5 appends timestamp, monitor, epoch, step
    # and floating loss to the checkpoint name, which otherwise exceeds the
    # Windows MAX_PATH limit in this deeply nested workspace.
    # The workspace temp root has a low-integrity write grant from the
    # launcher; keep one subdirectory per replica to avoid concurrent scvi
    # callbacks choosing the same timestamp directory.
    log_root = (Path(job['temp_root']) / f'model-{job["index"] + 1}') if os.name == 'nt' else output / 'scvi_log'
    log_root.mkdir(parents=True, exist_ok=True)
    scvi.settings.logging_dir = str(log_root)
    sys.path.insert(0, str(source / 'src'))
    from regvelo import REGVELOVI
    from regvelo._module import VELOVAE
    import regvelo._model as source_model

    torch.set_num_threads(job['threads'])
    torch.backends.cudnn.benchmark = False
    scvi.settings.seed = job['seed']
    torch.manual_seed(job['seed'])
    np.random.seed(job['seed'])
    data = read_observations(Path(job['observations']))
    genes = list(data['genes'])
    n_genes = len(genes)
    feature_order = list(data['featureOrder'])
    raw_features = np.array([c['raw'] for c in data['cells']], dtype=np.float32)
    s_indices = [feature_order.index(f'S_{gene}') for gene in genes]
    u_indices = [feature_order.index(f'U_{gene}') for gene in genes]
    raw = np.concatenate([raw_features[:, s_indices], raw_features[:, u_indices]], axis=1)
    spliced_raw = raw[:, :n_genes].copy()
    unspliced_raw = raw[:, n_genes:].copy()
    n = len(raw)
    adata = ad.AnnData(spliced_raw,
                       obs=pd.DataFrame(index=[c['id'] for c in data['cells']]),
                       var=pd.DataFrame(index=genes))
    adata.layers['Ms'] = spliced_raw
    adata.layers['Mu'] = unspliced_raw
    prior_meta = data.get('prior') or {}
    skeleton = prior_meta.get('skeleton')
    if skeleton is None:
        # Legacy website observations use a permissive all-off-diagonal prior.
        skeleton = (np.ones((n_genes, n_genes), dtype=np.float32)
                    - np.eye(n_genes, dtype=np.float32)).tolist()
    skeleton = np.asarray(skeleton, dtype=np.float32)
    if skeleton.shape != (n_genes, n_genes):
        raise ValueError(f'prior.skeleton has shape {skeleton.shape}; expected {(n_genes, n_genes)}.')
    # The source VELOVAE stores rows as targets and columns as regulators.
    adata.uns['skeleton'] = pd.DataFrame(skeleton, index=genes, columns=genes)
    regulators = job.get('regulators')
    REGVELOVI.setup_anndata(adata, spliced_layer='Ms', unspliced_layer='Mu')
    model = REGVELOVI(adata, W=torch.from_numpy(skeleton.copy()), regulators=regulators,
                      soft_constraint=job['soft_constraint'], lam=job['lam'], lam2=job['lam2'])
    # This local REGVELOVI wrapper forwards custom dimension kwargs twice.
    # Rebuild its SAME VELOVAE class with explicit dimensions, retaining the
    # wrapper's data-derived initial values. No RNA equations/loss are replaced.
    original = model.module
    torch.manual_seed(job['seed'])
    regulator_list = None if regulators is None else [index for index, gene in enumerate(genes)
                                                       if gene in regulators]
    model.module = VELOVAE(n_input=n_genes, n_latent=job['n_latent'], n_hidden=job['hidden'], n_layers=1,
                           regulator_list=regulator_list,
                           regulator_index=original.regulator_index,
                           target_index=original.target_index,
                           skeleton=torch.from_numpy(skeleton.copy()),
                           lam=job['lam'], lam2=job['lam2'],
                           vector_constraint=not job['simple_dynamics'],
                           bias_constraint=not job['simple_dynamics'],
                           soft_constraint=job['soft_constraint'],
                           alpha_unconstr_init=original.alpha_unconstr.detach().cpu().numpy(),
                           alpha_1_unconstr_init=original.alpha_1_unconstr.detach().cpu().numpy())
    model.n_latent = job['n_latent']
    module = model.module
    resume = output / 'weights.pt'
    if job['resume']:
        if not resume.is_file():
            raise FileNotFoundError(f'No completed model to fine-tune: {resume}')
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['observationSha256'] != digest(Path(job['observations'])):
            raise ValueError('Cannot resume with different observations.')
        module.load_state_dict(saved['state_dict'])

    s = torch.from_numpy(spliced_raw)
    u = torch.from_numpy(unspliced_raw)

    def latent():
        module.eval()
        with torch.inference_mode():
            out = module.inference(s, u)
            return {'mu': out['qz_m'].cpu().tolist(), 'variance': out['qz_v'].cpu().tolist()}

    initial = latent()
    scope = job['scope']
    def enabled(name):
        if scope == 'joint':
            return True
        if scope == 'encoder':
            return name.startswith('z_encoder.')
        if scope == 'time':
            return name.startswith('decoder.')
        if scope == 'beta':
            return 'beta_mean_unconstr' in name
        if scope == 'gamma':
            return 'gamma_mean_unconstr' in name
        if scope == 'grn':
            return name.startswith('v_encoder.fc1.')
        if scope == 'rates':
            return 'beta_mean_unconstr' in name or 'gamma_mean_unconstr' in name or name.startswith('v_encoder.fc1.')
        raise ValueError(f'Unknown scope {scope}')
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(enabled(name))

    split_order = np.random.default_rng(40519).permutation(n)
    n_validation = max(3, int(round(.15 * n)))
    splits = [split_order[n_validation:], split_order[:n_validation], np.array([], dtype=np.int64)]
    history = []
    # Keep a lightweight, source-grounded trajectory for the paper teaching page.
    # The webpage must replay real optimization states instead of interpolating
    # between only the initial and final matrices.  A 6x6 W every ten epochs is
    # small enough to store alongside each completed fit.
    w_snapshots = []

    def snapshot_weight(epoch: int):
        weight = module.v_encoder.fc1.weight.detach().cpu().tolist()
        w_snapshots.append({'epoch': int(epoch), 'W': weight})

    # Epoch zero is the actual initialized W before the first optimizer update.
    snapshot_weight(0)

    class History(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            # Frozen networks must also retain their BatchNorm buffers and
            # disable dropout during module-specific fine-tuning.
            if scope not in ('joint', 'encoder'):
                module.z_encoder.eval()
            if scope not in ('joint', 'time'):
                module.decoder.eval()

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            values = {}
            for key, value in trainer.callback_metrics.items():
                if hasattr(value, 'numel') and value.numel() == 1 and torch.isfinite(value):
                    values[str(key)] = float(value.detach().cpu())
            history.append({'epoch': int(trainer.current_epoch) + 1, **values})
            epoch = int(trainer.current_epoch) + 1
            if epoch % 10 == 0:
                snapshot_weight(epoch)
            atomic_json(output / 'progress.json', {'status': 'training', 'seed': job['seed'],
                                                   'index': job['index'], 'epoch': int(trainer.current_epoch) + 1,
                                                   'budget': job['epochs'], 'metrics': values})
    checkpoint = ModelCheckpoint(dirpath=str(output / 'checkpoints'), filename='best',
                                  monitor='elbo_validation', mode='min', save_top_k=1)
    # Inject only the identical external split into stock scvi's DataSplitter.
    # RegVelo's complete model, losses, ModifiedTrainingPlan and ODE remain intact.
    with patch.object(source_model, 'DataSplitter', partial(DataSplitter, external_indexing=splits)):
        model.train(max_epochs=job['epochs'], batch_size=n, train_size=len(splits[0]) / n,
                    early_stopping=job['early_stopping'], accelerator='cpu', devices=1,
                    callbacks=[History(), checkpoint], enable_progress_bar=False,
                    logger=False, num_sanity_val_steps=0, enable_checkpointing=True)
    # Capture the final state of this actual optimization trajectory before
    # Lightning restores the best validation checkpoint for the exported fit.
    # The teaching page can therefore replay either the ten-epoch snapshots or
    # this terminal state without relabeling a best-checkpoint W as a later epoch.
    final_training_epoch = max((r['epoch'] for r in history), default=job['epochs'])
    if not w_snapshots or w_snapshots[-1]['epoch'] != final_training_epoch:
        snapshot_weight(final_training_epoch)
    if checkpoint.best_model_path:
        saved = torch.load(checkpoint.best_model_path, map_location='cpu', weights_only=False)
        # Lightning training plans store the complete module under 'module.'.
        state = {key.removeprefix('module.'): value for key, value in saved['state_dict'].items()
                 if key.startswith('module.')}
        if not state:
            raise ValueError('The best checkpoint contains no RegVelo module state.')
        module.load_state_dict(state, strict=True)
    module.eval()
    q = latent()
    with torch.inference_mode():
        inference = module.inference(s, u)
        mu, variance = inference['qz_m'], inference['qz_v']
        gamma, beta, alpha1 = module._get_rates()
        def generate(z):
            generated = module.generative(z=z, gamma=gamma, beta=beta, alpha_1=alpha1)
            prediction = torch.cat([generated['dist_s'].mean, generated['dist_u'].mean], dim=1)
            return {'prediction': prediction.cpu().tolist(), 'time': generated['ind_t'].cpu().tolist()}
        mean_pass = generate(mu)
        # Same epsilon bank across runs permits controlled, same-epsilon comparisons.
        epsilons = np.random.default_rng(501923).standard_normal((job['samples'], n, job['n_latent'])).astype(np.float32)
        samples = []
        for epsilon in epsilons:
            z = mu + variance.sqrt() * torch.from_numpy(epsilon)
            samples.append({'epsilon': epsilon.tolist(), 'z': z.cpu().tolist(), **generate(z)})
        x = torch.cat([s, u], dim=-1)
        if module.log_variational:
            x = torch.log(.01 + x)
        hidden = module.z_encoder.encoder(x)
        decoder_hidden = module.decoder.rho_first_decoder(mu)
        mean_head = module.z_encoder.mean_encoder
        var_head = module.z_encoder.var_encoder
        time_head = module.decoder.px_rho_decoder[0]
        fc = module.v_encoder.fc1
        def array(tensor):
            return tensor.detach().cpu().tolist()
        encoder_info = {'input': array(x), 'hidden': array(hidden),
                        'meanWeight': array(mean_head.weight), 'meanBias': array(mean_head.bias),
                        'varianceWeight': array(var_head.weight), 'varianceBias': array(var_head.bias),
                        'varianceEpsilon': float(module.z_encoder.var_eps)}
        time_info = {'hiddenMu': array(decoder_hidden), 'weight': array(time_head.weight),
                     'bias': array(time_head.bias)}
        rates = {'beta': array(beta), 'gamma': array(gamma), 'W': array(fc.weight),
                 'b': array(fc.bias) if fc.bias is not None else [0.] * n_genes,
                 'alphaObserved': array(module.v_encoder.transcription_rate(s))}
    observation_sha = digest(Path(job['observations']))
    torch.save({'state_dict': module.state_dict(), 'seed': job['seed'],
                'observationSha256': observation_sha, 'config': job}, resume)
    result = {'index': job['index'], 'seed': job['seed'], 'implementation': 'complete-local-regvelo',
              'epochs': max((r['epoch'] for r in history), default=job['epochs']),
              'bestEpoch': int(saved.get('epoch', 0)) + 1 if checkpoint.best_model_path else job['epochs'],
              'observationSha256': observation_sha, 'initial': initial, **q,
              'meanPass': mean_pass, 'samples': samples, 'rates': rates,
              'encoder': encoder_info, 'timeDecoder': time_info,
              'history': history, 'wSnapshots': w_snapshots, 'snapshotEvery': 10,
              'split': {'train': splits[0].tolist(), 'validation': splits[1].tolist()},
              'source': {'module': digest(source / 'src/regvelo/_module.py'),
                         'model': digest(source / 'src/regvelo/_model.py'),
                         'runner': digest(Path(__file__))},
              'versions': {'torch': torch.__version__, 'scvi': scvi.__version__},
               'config': {'n_genes': n_genes, 'genes': genes, 'featureOrder': feature_order,
                          'n_latent': job['n_latent'], 'n_hidden': job['hidden'], 'n_layers': 1,
                          't_max': float(module.t_max), 'scope': scope,
                          'prior': 'provided-skeleton' if prior_meta.get('skeleton') is not None else 'legacy-all-off-diagonal',
                          'soft_constraint': job['soft_constraint'], 'lam': job['lam'],
                          'lam2': job['lam2'], 'simple_dynamics': job['simple_dynamics'],
                          'regulators': regulators,
                          'wrapperAdaptation': 'Same source VELOVAE rebuilt to avoid duplicate dimension kwargs in local REGVELOVI wrapper'}}
    atomic_json(output / 'result.json', result)
    atomic_json(output / 'progress.json', {'status': 'complete', 'epoch': result['epochs'],
                                           'index': job['index'], 'seed': job['seed']})
    return result


def publish(data, output: Path, observations: Path, model_indices: list[int]):
    models = []
    for index in model_indices:
        result = output / f'model-{index + 1}/result.json'
        if result.is_file():
            item = json.loads(result.read_text(encoding='utf-8'))
            if item['observationSha256'] != digest(observations):
                raise ValueError('Saved runs use different observations.')
            models.append(item)
    bundle = {'format': 'regvelo-replicas-v1', 'status': 'complete' if len(models) == len(model_indices) else 'partial',
              'generatedAt': datetime.now(timezone.utc).isoformat(),
              'expectedModelCount': len(model_indices), 'data': data, 'models': models,
              'provenance': {'training': 'complete local RegVelo, stock torchode and scvi TrainingPlan',
                             'data': f"same observations, {len(data['genes'])}-gene expression space",
                             'comparison': 'same observations, prior and external train/validation split; distinct seeds'}}
    atomic_json(output / 'replicas.json', bundle)
    text = 'export const regveloReplicas = ' + json.dumps(bundle, ensure_ascii=False, allow_nan=False) + ';\n'
    target = ROOT / 'dist/regvelo-results.mjs'
    temporary = target.with_suffix('.mjs.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(target)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--observations', type=Path, default=ROOT / '.runtime/observations.json')
    p.add_argument('--output', type=Path, default=ROOT / '.runtime/regvelo-runs')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--latent', type=int, default=3, help='Number of latent dimensions (default: 3).')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--models', default='1,2,3')
    p.add_argument('--seed', type=int, help='Override the seed for a single-model stability run.')
    p.add_argument('--no-publish', action='store_true', help='Keep this run isolated; do not replace the main dist result bundle.')
    p.add_argument('--scope', choices=['joint', 'encoder', 'time', 'rates', 'grn', 'beta', 'gamma'], default='joint')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-early-stopping', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--probe-imports', action='store_true', help='Also verify actual imports, including local RegVelo.')
    args = p.parse_args()
    status = preflight(args.source, args.probe_imports or not args.preflight)
    print(json.dumps(status, indent=2), flush=True)
    if args.preflight:
        return 0 if status['ready'] else 2
    if not status['ready']:
        print('Runtime is missing dependencies or failed imports. See preflight details. Version warnings alone do not block execution.', file=sys.stderr)
        return 2
    indices = [int(x) - 1 for x in args.models.split(',')]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(SEEDS) for i in indices):
        p.error(f'--models must be distinct values from 1,{len(SEEDS)}')
    if not 1 <= args.epochs <= 10000 or not 2 <= args.samples <= 100 or args.hidden < 8 or args.latent < 1:
        p.error('Invalid epoch/sample/hidden dimensions.')
    if args.threads < 1 or args.workers < 1:
        p.error('Threads and workers must be positive.')
    if args.resume and indices != [0]:
        p.error('Fine-tuning is limited to model 1, which supplies all explanations.')
    data = read_observations(args.observations)
    temp_root = ROOT / '.runtime/temp/scvi-logs'
    temp_root.mkdir(parents=True, exist_ok=True)
    model_settings = resolve_model_settings(data)
    jobs = [{'index': index, 'seed': args.seed if args.seed is not None else SEEDS[index], 'source': str(args.source.resolve()),
             'observations': str(args.observations.resolve()), 'output': str(args.output.resolve()),
             'temp_root': str(temp_root.resolve()),
             'epochs': args.epochs, 'samples': args.samples, 'hidden': args.hidden, 'n_latent': args.latent,
             **model_settings,
             'threads': args.threads, 'scope': args.scope, 'resume': args.resume,
             'early_stopping': not args.no_early_stopping} for index in indices]
    failed = False
    with ProcessPoolExecutor(max_workers=max(1, min(len(jobs), args.workers))) as pool:
        futures = {pool.submit(fit_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                print(f'Model {result["index"] + 1} complete; best epoch {result["bestEpoch"]}', flush=True)
            except Exception:
                failed = True
                error = traceback.format_exc()
                print(error, file=sys.stderr, flush=True)
                try:
                    atomic_json(args.output / f'model-{job["index"] + 1}/error.json', {'error': error, 'status': 'failed'})
                except OSError as write_error:
                    print(f'Cannot persist worker error: {write_error}', file=sys.stderr, flush=True)
    if not failed and not args.no_publish:
        try:
            publish(data, args.output, args.observations, indices)
        except OSError as publish_error:
            # Low-integrity Windows workers may be unable to replace dist files.
            print(f'Publish deferred to launcher: {publish_error}', file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
