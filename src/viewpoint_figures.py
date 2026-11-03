"""Локальные контактные листы и EDA зафиксированных выборок."""
from pathlib import Path
from collections import Counter
from PIL import Image,ImageOps,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.viewpoint_audit import read,write,folders


def sheets(root,rows,name):
    _,report,_,runtime=folders(root)
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),12)
    write(report/'figures'/f'{name}_index.csv',[dict(visual_index=i+1,**r) for i,r in enumerate(rows)])
    for start in range(0,len(rows),24):
        batch=rows[start:start+24];canvas=Image.new('RGB',(1000,192*((len(batch)+3)//4)),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(batch):
            x=(j%4)*250;y=(j//4)*192
            with Image.open(runtime/'cache'/(row['sha256']+'.png')) as source:
                thumb=ImageOps.contain(source.convert('RGB'),(246,140))
            canvas.paste(thumb,(x+(250-thumb.width)//2,y))
            label=f"{start+j+1}: {row['image_id']}\n{row['poi_id'].split('/')[-1]}\n{row['viewpoint']}"
            draw.text((x+2,y+141),label,fill='black',font=font)
        canvas.save(report/'figures'/f'{name}_{start//24+1:02}.jpg',quality=88)


def make_figures(root):
    root=Path(root);data,report,_,_=folders(root)
    for seed in [42,43,44]:
        for mode in ['O','R']:
            sheets(root,read(data/'manifests'/f'train_{mode}_seed{seed}.csv'),f'selected_{mode}_seed{seed}')
    labels=read(data/'viewpoint_labels.csv')
    sheets(root,[r for r in labels if r['split']=='train' and r['viewpoint']=='undetermined'],'undetermined')
    groups=[read(data/'manifests'/f'{name}.csv') for name in ['train_O_seed42','train_R_seed42','val','test']]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    pois=sorted({r['poi_id'] for r in groups[0]})
    for rows,label in zip(groups,['O train','R train','val','test']):
        counts=Counter(r['poi_id'] for r in rows)
        axes[0,0].plot(range(len(pois)),[counts[p] for p in pois],marker='o',label=label)
        ratios=[int(r['width'])/int(r['height']) for r in rows]
        axes[0,1].hist(ratios,bins=12,alpha=.35,label=label)
    axes[0,0].set_xticks(range(len(pois)),[p.split('/')[-1] for p in pois],rotation=55,ha='right',fontsize=8)
    axes[0,0].set_ylabel('images');axes[0,0].legend();axes[0,0].set_title('Class counts')
    axes[0,1].set_title('Aspect ratio');axes[0,1].legend()
    for mode,offset in [('O',-.18),('R',.18)]:
        count=Counter(r['viewpoint'] for r in groups[0 if mode=='O' else 1])
        axes[1,0].bar(np.arange(3)+offset,[count[v] for v in ['main','left','right']],width=.36,label=mode)
    axes[1,0].set_xticks(range(3),['main','left','right']);axes[1,0].legend();axes[1,0].set_title('Train viewpoints (seed42)')
    source_counts=Counter(r['source'] for r in labels)
    axes[1,1].bar(range(len(source_counts)),list(source_counts.values()))
    axes[1,1].set_xticks(range(len(source_counts)),list(source_counts),rotation=40,ha='right',fontsize=8)
    axes[1,1].set_title('Inventory provenance')
    fig.tight_layout();fig.savefig(report/'figures/eda.png',dpi=130);plt.close(fig)
