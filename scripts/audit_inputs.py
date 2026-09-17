"""Read every supplied document, metadata row and image; preserve original inputs."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'audit'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    for p in ROOT.glob('*.docx'):
        lines = []
        with ZipFile(p) as z:
            for name in z.namelist():
                if name.startswith('word/') and name.endswith('.xml') and any(k in name for k in ['document.xml', 'header', 'footer', 'footnotes', 'endnotes', 'comments']):
                    tree = ET.fromstring(z.read(name))
                    for para in tree.findall('.//w:p', ns):
                        text = ''.join(x.text or '' for x in para.findall('.//w:t', ns))
                        if text.strip(): lines.append(text)
                if name.startswith('word/media/'):
                    dest = OUT / (p.stem + '_' + Path(name).name)
                    dest.write_bytes(z.read(name))
        (OUT / (p.stem + '.txt')).write_text('\n'.join(lines), encoding='utf-8')
        manifest.append({'file':p.name, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    archive = next(ROOT.glob('DATASETS-*.zip'))
    manifest.append({'file':archive.name, 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    with ZipFile(archive) as z:
        for name in ['source_image_metadata.csv', 'crop_metadata_index.csv']:
            (data / name).write_bytes(z.read('DATASETS/' + name))
        payload = z.read('DATASETS/images.zip')
    sources = pd.read_csv(data / 'source_image_metadata.csv')
    index = pd.read_csv(data / 'crop_metadata_index.csv')
    assert sources.source_image_id.is_unique and index.filename.is_unique
    joined = index.merge(sources, on='source_image_id', how='left', validate='many_to_one', indicator=True)
    assert joined['_merge'].eq('both').all()
    modes, sizes, hashes = Counter(), Counter(), defaultdict(list)
    stats = []
    rng = np.random.default_rng(2026)
    chosen = set(rng.choice(index.filename, size=36, replace=False))
    sheet = Image.new('RGB', (6*190, 6*208), 'white')
    draw = ImageDraw.Draw(sheet)
    tiles = 0
    with ZipFile(io.BytesIO(payload)) as z:
        entries = [x for x in z.infolist() if not x.is_dir()]
        assert len(entries) == len(index)
        assert {Path(x.filename).name for x in entries} == set(index.filename)
        for entry in entries:
            name = Path(entry.filename).name
            raw = z.read(entry)
            image = Image.open(io.BytesIO(raw))
            image.load()
            sizes[str(image.size)] += 1
            modes[image.mode] += 1
            arr = np.asarray(image.convert('L'))
            hashes[hashlib.sha256(arr.tobytes()).hexdigest()].append(name)
            stats.append({'filename':name, 'mean':float(arr.mean()), 'std':float(arr.std()), 'min':int(arr.min()), 'max':int(arr.max()), 'zero_fraction':float((arr==0).mean()), 'white_fraction':float((arr==255).mean())})
            if name in chosen:
                x,y = (tiles%6)*190, (tiles//6)*208
                sheet.paste(image.convert('RGB').resize((184,184)), (x,y))
                draw.text((x+2,y+186),name,fill='black')
                tiles += 1
            dest = data / 'images' / name
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists(): dest.write_bytes(raw)
    frame = pd.DataFrame(stats)
    frame.to_csv(OUT / 'image_statistics.csv', index=False)
    duplicates = [v for v in hashes.values() if len(v)>1]
    summary = {'images':len(stats), 'sources':len(sources), 'sizes':dict(sizes), 'modes':dict(modes), 'missing_metadata':int(joined.isna().sum().sum()), 'duplicate_pixel_groups':duplicates, 'metadata_unique_values':{c:int(sources[c].nunique()) for c in sources}, 'metadata_description':sources.describe(include='all').fillna('').to_dict(), 'crops_per_source':index.groupby('source_image_id').size().describe().to_dict(), 'pixel_statistics':frame.describe().to_dict(), 'manifest':manifest}
    (OUT / 'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    sheet.save(OUT / 'random_batch.png')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['pixel_statistics','manifest','metadata_description','duplicate_pixel_groups']},indent=2))
    print('Duplicate pixel groups:',len(duplicates),'extra copies:',sum(len(v)-1 for v in duplicates))
    print(sources.describe(include='all').to_string())

if __name__ == '__main__': main()
