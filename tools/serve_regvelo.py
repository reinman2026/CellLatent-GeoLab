"""Serve the lab on localhost; run complete RegVelo through its isolated Python."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'
OUTPUT = RUNTIME / 'regvelo-runs'
ISOLATED_PYTHON = RUNTIME / 'regvelo-venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
SOURCE_PYTHON = Path('E:/PycharmProjects/rna_window_response_hlc_snapshot-regnet/ref/regvelo/localtest/.venv_regvelo/Scripts/python.exe')
PROCESS_FLAGS = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}


class TrainingService:
    def __init__(self, python=None):
        self.python = Path(python) if python else next((p for p in (ISOLATED_PYTHON, SOURCE_PYTHON) if p.is_file()), ISOLATED_PYTHON)
        self.lock = threading.Lock()
        self.process = None
        self.error = ''
        self.preflight = {'ready': False, 'reason': 'Isolated RegVelo environment is not installed.'}
        if self.python.is_file():
            try:
                result = subprocess.run([str(self.python), str(ROOT / 'tools/train_regvelo_replicas.py'), '--preflight', '--probe-imports'],
                                        capture_output=True, text=True, timeout=60, cwd=ROOT,
                                        env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                                             'NUMBA_DISABLE_JIT': '1'}, **PROCESS_FLAGS)
                self.preflight = json.loads(result.stdout)
            except (OSError, subprocess.TimeoutExpired, ValueError) as error:
                self.preflight = {'ready': False, 'reason': str(error)}

    def status(self):
        with self.lock:
            progress = []
            for index in range(1, 4):
                path = OUTPUT / f'model-{index}/progress.json'
                if path.is_file():
                    try:
                        progress.append(json.loads(path.read_text(encoding='utf-8')))
                    except (OSError, ValueError):
                        pass
            return {'ready': self.preflight.get('ready', False),
                    'running': self.process is not None and self.process.poll() is None,
                    'hasResults': (OUTPUT / 'replicas.json').is_file(),
                    'progress': progress, 'error': self.error, 'preflight': self.preflight}

    def start(self, request):
        with self.lock:
            if not self.preflight.get('ready'):
                raise ValueError('完整 RegVelo 环境尚未通过依赖、版本与实际导入检查；查看 /api/status 的 preflight 详情。')
            if self.process is not None and self.process.poll() is None:
                raise ValueError('已有训练任务正在运行。')
            scope = request.get('scope', 'joint')
            all_models = request.get('all', False)
            epochs = request.get('epochs', 1000)
            if scope not in ('joint', 'encoder', 'time', 'rates', 'grn', 'beta', 'gamma'):
                raise ValueError('未知训练模块。')
            if type(all_models) is not bool or type(epochs) is not int or not 1 <= epochs <= 10000:
                raise ValueError('训练轮数需要在 1 至 10000 之间。')
            if all_models and scope != 'joint':
                raise ValueError('三个模型的独立训练必须使用联合模式。')
            if not all_models and not (OUTPUT / 'model-1/weights.pt').is_file():
                raise ValueError('请先完成模型 1 训练，再微调该模型。')
            command = [str(self.python), str(ROOT / 'tools/train_regvelo_replicas.py'),
                       '--epochs', str(epochs), '--scope', scope,
                       '--models', '1,2,3' if all_models else '1']
            if not all_models:
                command.append('--resume')
            OUTPUT.mkdir(parents=True, exist_ok=True)
            log = (OUTPUT / 'training.log').open('w', encoding='utf-8')
            try:
                self.process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                                                     'NUMBA_DISABLE_JIT': '1'}, **PROCESS_FLAGS)
            except Exception:
                log.close()
                raise
            self.error = ''
            process = self.process

            def finish():
                code = process.wait()
                log.close()
                with self.lock:
                    if code:
                        self.error = f'Python 训练退出码 {code}；查看 .runtime/regvelo-runs/training.log。'
                if code == 0:
                    subprocess.run(['node', str(ROOT / 'tools/build-offline.mjs')], cwd=ROOT,
                                   capture_output=True, **PROCESS_FLAGS)
            threading.Thread(target=finish, daemon=True).start()
            return {'started': True, 'models': [1, 2, 3] if all_models else [1], 'scope': scope}


class LabHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, service, **kwargs):
        self.service = service
        super().__init__(*args, directory=str(ROOT / 'dist'), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def json_response(self, value, status=200):
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def valid_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',
                                            f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self.valid_host():
            return self.json_response({'error': 'Invalid local host'}, 403)
        if self.path == '/api/status':
            return self.json_response(self.service.status())
        if self.path == '/api/results':
            path = OUTPUT / 'replicas.json'
            if not path.is_file():
                return self.json_response({'error': 'No fitted results'}, 404)
            return self.json_response(json.loads(path.read_text(encoding='utf-8')))
        return super().do_GET()

    def do_POST(self):
        origin = self.headers.get('Origin', '')
        valid_origins = (f'http://127.0.0.1:{self.server.server_port}',
                         f'http://localhost:{self.server.server_port}')
        if not self.valid_host() or origin not in valid_origins:
            return self.json_response({'error': 'Same-origin localhost requests required'}, 403)
        if self.path != '/api/train':
            return self.json_response({'error': 'Unknown endpoint'}, 404)
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.json_response({'error': 'JSON required'}, 415)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 1 <= size <= 4096:
                raise ValueError('Invalid request size')
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError('JSON object required')
            return self.json_response(self.service.start(request))
        except (ValueError, OSError) as error:
            return self.json_response({'error': str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--python', type=Path, help='Python interpreter containing compatible RegVelo dependencies.')
    args = parser.parse_args()
    service = TrainingService(args.python)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(LabHandler, service=service))
    print(f'Open http://127.0.0.1:{server.server_port}/#replicas', flush=True)
    print(json.dumps(service.preflight, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
