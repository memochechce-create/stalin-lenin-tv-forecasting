"""Refresh saved vector figure labels without recalculating forecast data.

Usage: python analysis/refresh_saved_figure_labels.py --source-dir PATH
Saved SVGs are cached outputs of project_figures.py. Only text nodes change;
chart marks, dates, counts, intervals and coordinates are preserved.
"""
import argparse,json,re
from pathlib import Path
import xml.etree.ElementTree as ET

DASHES=str.maketrans({chr(n):'-' for n in range(0x2012,0x2016)})
ET.register_namespace('','http://www.w3.org/2000/svg')
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')

def refresh(source_dir,output_dir,lang):
    mapping=json.loads(Path(__file__).with_name('figure_label_translations.json').read_text(encoding='utf-8'))
    output_dir.mkdir(parents=True,exist_ok=True)
    done=[]
    for src in sorted(source_dir.glob('*.svg')):
        raw=src.read_text(encoding='utf-8').translate(DASHES)
        root=ET.fromstring(raw)
        for node in root.iter():
            if node.tag.rsplit('}',1)[-1]!='text' or not node.text:continue
            label=node.text
            if lang=='en':
                if label in mapping:label=mapping[label]
                else:
                    for ru,en in sorted(mapping.items(),key=lambda x:len(x[0]),reverse=True):
                        # Single words may only replace complete label tokens.
                        if re.fullmatch(r'\w+',ru):label=re.sub(r'(?<!\w)'+re.escape(ru)+r'(?!\w)',lambda _:en,label)
                        else:label=label.replace(ru,en)
                label=re.sub(r'(?<=\d),(?=\d)', '.',label)
                if re.search('[А-Яа-яЁё]',label):raise ValueError(f'{src.name}: untranslated label {label!r}')
            node.text=label.translate(DASHES)
        target=output_dir/src.name
        ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)
        assert not any(chr(n) in target.read_text(encoding='utf-8') for n in range(0x2012,0x2016))
        done.append(src.name)
    return done

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--lang',choices=['ru','en'],default='en')
    a=parser.parse_args();print(json.dumps(refresh(a.source_dir,a.output_dir,a.lang)))
