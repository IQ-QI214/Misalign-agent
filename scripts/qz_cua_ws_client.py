#!/usr/bin/env python3
"""SSH ProxyCommand over a binary WebSocket. Python standard library only."""
import base64
import hashlib
import os
from pathlib import Path
import socket
import select
import ssl
import struct
import sys
import time
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
    url = Path(sys.argv[1]).read_text().strip()
    for attempt in range(3):
        try:
            sock = connect(url)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)

    sock.setblocking(False)
    incoming = bytearray()
    outgoing = []
    send_needs_read = False
    receive_needs_write = False

    def enqueue(opcode, data):
        size = len(data)
        if size < 126:
            length = bytes([size | 128])
        elif size < 65536:
            length = b'\xfe' + struct.pack('!H', size)
        else:
            length = b'\xff' + struct.pack('!Q', size)
        mask = os.urandom(4)
        packet = bytes([128 | opcode]) + length + mask + bytes(x ^ mask[i % 4] for i, x in enumerate(data))
        if opcode == 10 and outgoing:
            outgoing.insert(1, packet)
        else:
            outgoing.append(packet)

    try:
        while True:
            reads = [sock]
            if len(outgoing) < 16:
                reads.append(0)
            writes = [sock] if (outgoing and not send_needs_read) or receive_needs_write else []
            readable, writable, _ = select.select(reads, writes, [], 1)
            if 0 in readable:
                data = os.read(0, 65536)
                if not data:
                    return
                enqueue(2, data)
            if outgoing and (sock in writable or send_needs_read and sock in readable):
                try:
                    count = sock.send(outgoing[0])
                    outgoing[0] = outgoing[0][count:]
                    if not outgoing[0]:
                        outgoing.pop(0)
                    send_needs_read = False
                except ssl.SSLWantReadError:
                    send_needs_read = True
                except ssl.SSLWantWriteError:
                    pass
            if sock in readable or sock.pending() or receive_needs_write and sock in writable:
                try:
                    data = sock.recv(65536)
                    if not data:
                        return
                    incoming.extend(data)
                    receive_needs_write = False
                except ssl.SSLWantReadError:
                    pass
                except ssl.SSLWantWriteError:
                    receive_needs_write = True
            while len(incoming) >= 2:
                first, second = incoming[:2]
                opcode, size, offset = first & 15, second & 127, 2
                if size in (126, 127):
                    width = 2 if size == 126 else 8
                    if len(incoming) < 2 + width:
                        break
                    size = int.from_bytes(incoming[2:2 + width], 'big')
                    offset += width
                if size > 4 * 1024 * 1024:
                    raise RuntimeError('Oversized WebSocket frame')
                masked = bool(second & 128)
                header_end = offset + (4 if masked else 0)
                if len(incoming) < header_end + size:
                    break
                mask = incoming[offset:header_end]
                data = bytes(incoming[header_end:header_end + size])
                del incoming[:header_end + size]
                if masked:
                    data = bytes(x ^ mask[i % 4] for i, x in enumerate(data))
                if opcode in (0, 2):
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                elif opcode == 9:
                    enqueue(10, data)
                elif opcode == 8:
                    if len(data) >= 2:
                        print('WebSocket close code=' + str(int.from_bytes(data[:2], 'big')), file=sys.stderr)
                    return
                elif opcode != 10:
                    raise RuntimeError('Expected binary SSH transport')
    finally:
        sock.close()


if __name__ == '__main__':
    main()
