import glob, os, shutil
from pathlib import Path

to_ann = 'data/to_annotate'

# Find ALL images anywhere in subfolders n, nm, oop
new_imgs = (
    glob.glob(f'{to_ann}/n/**/*.jpg',   recursive=True) +
    glob.glob(f'{to_ann}/n/**/*.png',   recursive=True) +
    glob.glob(f'{to_ann}/nm/**/*.jpg',  recursive=True) +
    glob.glob(f'{to_ann}/nm/**/*.png',  recursive=True) +
    glob.glob(f'{to_ann}/oop/**/*.jpg', recursive=True) +
    glob.glob(f'{to_ann}/oop/**/*.png', recursive=True)
)

print(f'New images found in n/nm/oop: {len(new_imgs)}')

# Find their matching labels
matched = []
for img in new_imgs:
    lbl = str(Path(img).with_suffix('.txt'))
    if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
        matched.append((img, lbl))
        print(f'  GOOD: {Path(img).name}  +  {Path(lbl).name}')
    else:
        print(f'  SKIP: {Path(img).name}  (no label or empty)')

print(f'Matched non-empty pairs: {len(matched)}')