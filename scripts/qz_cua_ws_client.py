#!/usr/bin/env python3
"""SSH ProxyCommand over a binary WebSocket. Python standard library only."""
import base64
import hashlib
import os
from pathlib import Path
import socket
import ssl
import struct
import sys
import threading
from urllib.parse import urlsplit


def headers(sock):
    data = bytearray()
    while not data.endswith(b'\r\n\r\n'):
        chunk = sock.recv(1)
        if not chunk or len(data) >= 32768:
            raise RuntimeError('Incomplete or oversized HTTP response')
        data.extend(chunk)
    lines = data.decode('iso-8859-1').split('\r\n')
    fields = [line.split(':', 1) for line in lines[1:] if ':' in line]
    return lines[0], {name.lower(): value for name, value in fields}


def connect(url):
    endpoint = urlsplit(url)
    if endpoint.scheme != 'https' or not endpoint.hostname or not endpoint.hostname.endswith('.sii.edu.cn'):
        raise ValueError('Expected the QZ HTTPS port proxy')
    host = endpoint.hostname
    proxy = urlsplit(os.environ.get('HTTPS_PROXY', ''))
    if proxy.hostname:
        sock = socket.create_connection((proxy.hostname, proxy.port or 80), timeout=15)
        sock.sendall(f'CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n'.encode())
        status, _ = headers(sock)
        if ' 200 ' not in status:
            raise RuntimeError('HTTPS proxy CONNECT failed')
    else:
        sock = socket.create_connection((host, 443), timeout=15)
    sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode()
    path = endpoint.path or '/'
    if endpoint.query:
        path += '?' + endpoint.query
    sock.sendall((f'GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n'
                  f'Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
    status, response = headers(sock)
    expected = base64.b64encode(hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
    if ' 101 ' not in status or response.get('sec-websocket-accept', '').strip() != expected:
        raise RuntimeError('QZ WebSocket upgrade failed: ' + status)
    sock.settimeout(None)
    return sock


def main():
    sock = connect(Path(sys.argv[1]).read_text().strip())
    lock = threading.Lock()

    def send(opcode, data):
        size = len(data)
        if size < 126:
            length = bytes([size | 128])
        elif size < 65536:
            length = b'\xfe' + struct.pack('!H', size)
        else:
            length = b'\xff' + struct.pack('!Q', size)
        mask = os.urandom(4)
        packet = bytes([128 | opcode]) + length + mask + bytes(x ^ mask[i % 4] for i, x in enumerate(data))
        with lock:
            sock.sendall(packet)

    def exact(size):
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise EOFError()
            data.extend(chunk)
        return bytes(data)

    def input_loop():
        try:
            while True:
                data = os.read(0, 65536)
                if not data:
                    send(8, b'')
                    return
                send(2, data)
        except OSError:
            sock.close()

    threading.Thread(target=input_loop, daemon=True).start()
    try:
        while True:
            first, second = exact(2)
            opcode, size = first & 15, second & 127
            if size == 126:
                size = struct.unpack('!H', exact(2))[0]
            elif size == 127:
                size = struct.unpack('!Q', exact(8))[0]
            if size > 4 * 1024 * 1024:
                raise RuntimeError('Oversized WebSocket frame')
            mask = exact(4) if second & 128 else None
            data = exact(size)
            if mask:
                data = bytes(x ^ mask[i % 4] for i, x in enumerate(data))
            if opcode in (0, 2):
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            elif opcode == 9:
                send(10, data)
            elif opcode == 8:
                return
            elif opcode != 10:
                raise RuntimeError('Expected binary SSH transport')
    except EOFError:
        return
    finally:
        sock.close()


if __name__ == '__main__':
    main()
