# Quickstart: Multi-Session Router

## 1. Run Multi-Session Server (Mock Mode, No GPU)

```bash
# Start a worker with mock backend (no model download needed)
python -m moshi.server --mock --max-sessions 3 --port 8998

# In another terminal, test with multiple WebSocket clients:
python -c "
import asyncio, aiohttp

async def test_client(port, name):
    async with aiohttp.ClientSession() as s:
        ws = await s.ws_connect(
            f'http://localhost:{port}/api/chat?voice_prompt=mock&text_prompt=test'
        )
        msg = await ws.receive_bytes()
        assert msg == b'\x00', f'{name}: expected handshake'
        print(f'{name}: connected and received handshake')
        await ws.close()
        print(f'{name}: disconnected')

async def main():
    await asyncio.gather(
        test_client(8998, 'client-1'),
        test_client(8998, 'client-2'),
        test_client(8998, 'client-3'),
    )

asyncio.run(main())
"
```

Expected output:
```
client-1: connected and received handshake
client-2: connected and received handshake
client-3: connected and received handshake
client-1: disconnected
client-2: disconnected
client-3: disconnected
```

## 2. Run with Router (Mock Mode)

```bash
# Terminal 1: Worker A
python -m moshi.server --mock --max-sessions 2 --port 8998 --worker-id worker-a

# Terminal 2: Worker B
python -m moshi.server --mock --max-sessions 2 --port 8999 --worker-id worker-b

# Terminal 3: Router
python -m moshi.router \
  --port 9000 \
  --workers worker-a:localhost:8998 \
  --workers worker-b:localhost:8999
```

Check router health:
```bash
curl http://localhost:9000/health
# {"total_workers": 2, "healthy_workers": 2, "total_capacity": 4, "active_sessions": 0, "available_slots": 4}
```

Check worker metrics:
```bash
curl http://localhost:8998/metrics
# personaplex_active_sessions 0
```

## 3. Run Automated Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio

# Run all tests (no GPU required)
pytest tests/ -v

# Run only session concurrency tests
pytest tests/test_session_manager.py -v

# Run only router tests
pytest tests/test_router.py -v
```

## 4. Run with Real Model (GPU)

```bash
# Single worker with multiple sessions
HF_TOKEN=<TOKEN> python -m moshi.server \
  --max-sessions 3 \
  --port 8998 \
  --worker-id gpu-worker-1

# Connect via WebUI at https://localhost:8998
# Multiple browser tabs can now connect simultaneously
```

## 5. Verify Session Isolation

```bash
# Start mock server and run isolation test
python -m moshi.server --mock --max-sessions 2 --port 8998

pytest tests/test_session.py::test_session_state_isolation -v
```

This test connects two clients with different voice prompts and
seeds, then verifies that each session's configuration is
independent and that disconnecting one does not affect the other.
