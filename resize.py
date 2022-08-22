#!/usr/bin/env python3
import json
import os
import sys
from typing import Dict, Sequence

from PIL import Image, ImageOps


def main(argv: Sequence[str] = ()) -> None:
    source_dir = argv[0]
    output_dir = argv[1]
    
    print(f'Resizing files from {source_dir} to {output_dir}', file=sys.stderr)

    for root, dirs, files in os.walk(source_dir):
        for entry in files:
            source_object_path = os.path.join(root, entry)
            orig_image = Image.open(source_object_path)
            prefix, ext = os.path.splitext(entry)
            
            if ext.lstrip("."):
                fmt = None
            else:
                fmt = orig_image.format or "PNG"
                ext = ""
            
            print(f'Processing {source_object_path}', file=sys.stderr)
            new_image = ImageOps.fit(orig_image, (256, 256))
            dest_object_path = os.path.join(output_dir, entry)
            rgb_im = new_image.convert('RGB')
            print(f'Saving {dest_object_path}', file=sys.stderr)
            rgb_im.save(dest_object_path, format=fmt)


if __name__ == "__main__":
    main(json.loads(sys.stdin.read()))
