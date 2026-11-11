"""Open the paired RT11 robot's tuning panel once per lap on the operator's Mac."""
import hashlib
import hmac
import json
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import urlsplit
import secrets
import select
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import build_opener, ProxyHandler


def signature(key, message):
    return hmac.new(key, message.encode(), hashlib.sha256).hexdigest()


def session_id(advert):
    value = advert['session']
    if len(value) != 32 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid RT11 session')
    return value


def listen():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(('', 18762))
    except OSError:
        sock.close()
        return None
    return sock


LOCAL_PORT = 18760
LOCAL_URL = 'http://127.0.0.1:' + str(LOCAL_PORT)
WAIT_PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="2;url=/local"><title>RT11 箱区调参</title>
<body style="font:18px/1.6 system-ui;padding:3em;background:#f1f5f1;color:#153a32">
<h1>等待 RT11 启动</h1><p>保持这个页面打开。机器人面板上线后会自动连接。</p></body></html>""".encode()


def current_session(client, base, key):
    nonce = secrets.token_hex(16)
    with client.open(base + '/session?nonce=' + nonce, timeout=1.2) as response:
        advert = json.loads(response.read(2048))
    session = session_id(advert)
    if not hmac.compare_digest(advert['proof'], signature(key, 'open/' + session + '/' + nonce)):
        raise ValueError('Unpaired robot')
    return session


class LocalPanel(BaseHTTPRequestHandler):
    """Only relay RT11 panel routes. The browser sends its own per-lap token."""
    def log_message(self, *args):
        pass

    def reply(self, status, body=b'', mime='application/json', location=None):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        if location:
            self.send_header('Location', location)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def handle_panel(self):
        if self.headers.get('Host') not in ('127.0.0.1:' + str(LOCAL_PORT), 'localhost:' + str(LOCAL_PORT)):
            return self.reply(403)
        method = self.command
        allowed = ('/', '/api/state', '/local') if method == 'GET' else ('/api/settings',)
        if self.path not in allowed:
            return self.reply(404)
        base = self.server.robot_base
        try:
            if self.path == '/local':
                session = current_session(self.server.client, base, self.server.key)
                token = signature(self.server.key, 'panel/' + session)
                return self.reply(302, location='/#' + token)
            body = None
            headers = {}
            if self.path.startswith('/api/'):
                headers['Authorization'] = self.headers.get('Authorization', '')
            if method == 'POST':
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 2048:
                    return self.reply(413)
                if self.headers.get_content_type() != 'application/json':
                    return self.reply(415)
                headers['Content-Type'] = 'application/json'
                body = self.rfile.read(size)
            remote = urlsplit(base)
            upstream = HTTPConnection(remote.hostname, remote.port, timeout=2)
            try:
                upstream.request(method, self.path, body=body, headers=headers)
                response = upstream.getresponse()
                return self.reply(response.status, response.read(256 * 1024),
                                  response.getheader('Content-Type', 'application/json'))
            finally:
                upstream.close()
        except (OSError, HTTPException, ValueError, KeyError, TypeError):
            if self.path in ('/local', '/'):
                return self.reply(200, WAIT_PAGE, 'text/html; charset=utf-8')
            return self.reply(503, b'{}')

    do_GET = handle_panel
    do_POST = handle_panel


def watch():
    folder = Path(__file__).resolve().parent
    address = folder / 'robot-url'
    base = address.read_text().strip()
    key = (folder / 'pair.key').read_bytes()
    client = build_opener(ProxyHandler({}))
    server = ThreadingHTTPServer(('127.0.0.1', LOCAL_PORT), LocalPanel)
    server.robot_base = base
    server.key = key
    server.client = client
    Thread(target=server.serve_forever, daemon=True).start()
    marker = folder / 'last-local-session'
    last = marker.read_text() if marker.exists() else ''
    sock = listen()
    while True:
        try:
            if sock and select.select([sock], [], [], 2)[0]:
                packet, peer = sock.recvfrom(1024)
                advert = json.loads(packet)
                proof = signature(key, 'announce/' + session_id(advert))
                if hmac.compare_digest(advert['proof'], proof):
                    base = 'http://' + peer[0] + ':18761'
            elif sock is None:
                time.sleep(2)
            session = current_session(client, base, key)
            server.robot_base = base
            if session == last:
                continue
            subprocess.run(['/usr/bin/open', LOCAL_URL + '/local'], check=True, timeout=8)
            marker.write_text(session)
            address.write_text(base)
            last = session
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            continue


if __name__ == '__main__':
    watch()
