#!/usr/bin/env python3
"""Single-client HF inference worker over a QZ shared-directory mailbox."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import signal
import socket
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--mailbox', type=Path, required=True)
    parser.add_argument('--max-requests', type=int, default=20)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--idle-timeout', type=float, default=300)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('QZ_LINUX_REQUIRED')
    def stop_handler(signum, frame):
        raise SystemExit('Worker stopped by owner')
    signal.signal(signal.SIGTERM, stop_handler)
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForImageTextToText

    def write(name, value):
        target = args.mailbox / name
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False) + '\n')
        temporary.replace(target)

    try:
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa').to('cuda:0').eval()
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).eval()
        write('ready.json', {'model_run': True, 'hostname': socket.gethostname(), 'torch': torch.__version__, 'transformers': transformers.__version__,
                             'device': torch.cuda.get_device_name(0), 'model': str(args.model),
                             'started_utc': datetime.now(timezone.utc).isoformat()})
        from cua_defense.inference import serve_requests
        serve_requests(model, processor, args.mailbox, max_requests=args.max_requests,
                       idle_timeout=args.idle_timeout)
    except Exception:
        write('error.json', {'error': traceback.format_exc()})
        raise
    finally:
        write('stopped.json', {'time_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
