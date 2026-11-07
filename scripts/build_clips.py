"""Cut each episode's camera video into a small H.264 clip the Live page serves itself.

The Hub stores one long AV1 file per camera; browsers without AV1 decoding (and slow seeks into a
several-hundred-MB remote file) left the Live page with no video. H.264 plays everywhere.

Needs ffmpeg with an AV1 decoder and libx264 (e.g. `pip install imageio-ffmpeg`; `brew install ffmpeg`).
Usage:  python scripts/build_clips.py data/so100 [episode ...]
Writes app/static/clips/{cam}/ep{NNN}.mp4 (served at app/static/clips/... with server.enableStaticServing).
"""
import glob, json, os, shutil, subprocess, sys
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "data/so100"
only = {int(x) for x in sys.argv[2:]}
ff = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
if not ff:
    try:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found: install it, `pip install imageio-ffmpeg`, or set FFMPEG=/path/to/ffmpeg")

info = json.load(open(f"{root}/meta/info.json"))
em = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True))]).set_index("episode_index")
cams = sorted({c.split("/")[1] for c in em.columns if c.startswith("videos/") and c.endswith("/from_timestamp")})
for cam in cams:
    out_dir = f"app/static/clips/{cam.split('.')[-1]}"; os.makedirs(out_dir, exist_ok=True)
    for idx, r in em.iterrows():
        if only and idx not in only:
            continue
        src = os.path.join(root, info["video_path"].format(video_key=cam, chunk_index=int(r[f"videos/{cam}/chunk_index"]),
                                                            file_index=int(r[f"videos/{cam}/file_index"])))
        a, b = float(r[f"videos/{cam}/from_timestamp"]), float(r[f"videos/{cam}/to_timestamp"])
        dst = f"{out_dir}/ep{idx:03d}.mp4"
        subprocess.run([ff, "-v", "error", "-y", "-ss", f"{a:.4f}", "-i", src, "-t", f"{b - a:.4f}", "-an",
                        "-vf", "scale=480:-2", "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p",
                        "-g", "30", "-movflags", "+faststart", dst], check=True)
        print(dst, os.path.getsize(dst) // 1024, "KB")
