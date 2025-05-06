from datasets import load_dataset
import soundfile as sf
import os

# 设置目标时长（秒），例如 1800 秒 = 30 分钟
target_sec = 1800

# 加载 Common Voice 中文数据集
dataset = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "zh-CN",
    split="train",
    streaming=True,          # 边下边用，取够就停止
    trust_remote_code=True   # 自动运行自定义加载逻辑，无需交互确认
)

# 创建存储目录
demo_dir = "data/raw/asr/commonvoice_demo"
wav_dir = os.path.join(demo_dir, "wav")
os.makedirs(wav_dir, exist_ok=True)

# 提取子集并保存音频和转写
total = 0.0
with open(os.path.join(demo_dir, "transcripts.txt"), "w", encoding="utf-8") as fout:
    for i, item in enumerate(dataset):
        wav = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        duration = len(wav) / sr
        # 保存 wav 文件
        out_path = os.path.join(wav_dir, f"{i:04d}.wav")
        sf.write(out_path, wav, sr)
        # 保存转写文本
        sentence = item["sentence"].strip().lower()
        fout.write(f"{i:04d}|{sentence}\n")
        total += duration
        # 达到目标时长后停止
        if total >= target_sec:
            break

print(f"已提取 {i+1} 条样本，累计时长约 {total/60:.2f} 分钟")
