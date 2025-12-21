from pathlib import Path

# 動画フォルダ（ユーザ環境）
video_dir = Path('../movie/側転斜状')
out_list = Path('examples/extract_sideflip_skeleton/sideflip.list')

video_dir.mkdir(parents=False, exist_ok=True)  # 存在確認用（無ければエラー回避）
out_list.parent.mkdir(parents=True, exist_ok=True)

mp4s = sorted(video_dir.glob('*.mp4'))
with out_list.open('w', encoding='utf-8') as f:
    for p in mp4s:
        f.write(f"{p.resolve()} 1\n")  # フルパス と ラベル 1
print(f"Wrote {len(mp4s)} lines to {out_list}")