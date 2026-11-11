"""Regression for the case-sensitive WebSocket handshake accept value."""
import importlib.util
from pathlib import Path
import socket
import unittest

SPEC = importlib.util.spec_from_file_location('ws_client', Path(__file__).parents[1] / 'scripts/qz_cua_ws_client.py')
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


class HandshakeHeaders(unittest.TestCase):
    def test_header_names_normalized_and_accept_value_preserved(self):
        sender, receiver = socket.socketpair()
        try:
            sender.sendall(b'HTTP/1.1 101 Switching Protocols\r\nSec-WebSocket-Accept: AbC+/123=\r\n\r\n\x82')
            status, fields = CLIENT.headers(receiver)
            self.assertEqual(status, 'HTTP/1.1 101 Switching Protocols')
            self.assertEqual(fields['sec-websocket-accept'].strip(), 'AbC+/123=')
            self.assertEqual(receiver.recv(1), b'\x82')
        finally:
            sender.close()
            receiver.close()


if __name__ == '__main__':
    unittest.main()
