#!/usr/bin/env python3
import json
import sys
import multiprocessing
from typing import Dict, Sequence
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps


def main(argv: Sequence[str] = ()) -> None:
    source_dir = argv[0]
    output_dir = argv[1]
    
    print(f'Resizing files from {source_dir} to {output_dir}', file=sys.stderr)
    num_threads = multiprocessing.cpu_count() - 1
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for p in Path(source_dir).rglob('*'):
            if not p.is_dir():
                executor.submit(resize, source_dir, output_dir, p)
            

def resize(source_dir: str, output_dir: str, p: Path) -> None:
    print(f'Processing {p}', file=sys.stderr)
    orig_image = Image.open(p)
    prefix, ext = p.stem, p.suffix

    if ext.lstrip("."):
        fmt = None
    else:
        fmt = orig_image.format or "PNG"
        ext = ""

    new_image = ImageOps.fit(orig_image, (256, 256))
    dest_object_path = output_dir / p.relative_to(source_dir)
    dest_object_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_im = new_image.convert('RGB')
    print(f'Saving {dest_object_path}', file=sys.stderr)
    rgb_im.save(dest_object_path, format=fmt)


if __name__ == "__main__":
    main(json.loads(sys.stdin.read()))
