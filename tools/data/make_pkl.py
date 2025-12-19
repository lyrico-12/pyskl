# 任意の場所で実行（例: examples/extract_sideflip_skeleton/make_final_pkl.py）
from mmcv import load, dump
annotations = load('examples/extract_sideflip_skeleton/sideflip_annos.pkl')
# 全て train に入れる例（ラベルは既に 1）
split = {'train': [p.stem for p in __import__('pathlib').Path('/home/denjo/univ/AI_excersize/自由課題/movie/側転斜状').glob('*.mp4')],
         'test': []}
dump(dict(split=split, annotations=annotations), 'examples/extract_sideflip_skeleton/sideflip_hrnet.pkl')
print("wrote sideflip_hrnet.pkl")