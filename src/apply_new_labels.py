import glob, os
from pathlib import Path

to_ann = 'data/to_annotate'

for folder in ['n', 'nm', 'oop']:
    txts = glob.glob(f'{to_ann}/{folder}/**/*.txt', recursive=True)
    txts = [t for t in txts 
            if Path(t).name not in ('obj.names','obj.data',
                                     'Train.txt','train.txt')]
    print(f'Folder {folder}: {len(txts)} label files')
    for t in txts[:3]:
        size = os.path.getsize(t)
        print(f'  {Path(t).name}  ({size} bytes)')