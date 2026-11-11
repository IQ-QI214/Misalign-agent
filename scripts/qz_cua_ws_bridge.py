#!/usr/bin/env python3
"""WebSocket transport to the existing localhost SSH service for QZ port proxy.

Authentication and encryption are performed by SSH. No arbitrary TCP targets.
Server dependency: websockets==15.0.1, installed only on QZ.
"""
import asyncio
import contextlib


async def bridge(websocket):
    reader, writer = await asyncio.wait_for(asyncio.open_connection('127.0.0.1', 22224), timeout=5)

    async def from_ssh():
        while True:
            data = await reader.read(65536)
            if not data:
                return
            await websocket.send(data)

    async def to_ssh():
        async for data in websocket:
            if not isinstance(data, bytes):
                await websocket.close(code=1003, reason='Binary SSH transport required')
                return
            writer.write(data)
            await writer.drain()

    tasks = [asyncio.create_task(from_ssh()), asyncio.create_task(to_ssh())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
        await websocket.close()


async def main():
    from websockets.asyncio.server import serve

    async with serve(bridge, '0.0.0.0', 30003, compression=None, max_size=1048576):
        print('CUA_WS_READY port=30003 target=127.0.0.1:22224', flush=True)
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())
