import os
import argparse
import logging
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
from jiwer import wer
from TTS.api import TTS

# ---------------------------
# 多语言支持映射表（英、法、德）
# ---------------------------
TGT_LANG_MAP = {
    "en": "en",
    "fr": "fr",
    "de": "de",
}

LANG_TO_TTS_MODEL = {
    "en": "tts_models/en/ljspeech/tacotron2-DDC",
    "fr": "tts_models/fr/css10/vits",
    "de": "tts_models/de/thorsten/tacotron2-DCA",
}

# ---------------------------
# CLI 参数解析
# ---------------------------
parser = argparse.ArgumentParser(description="ASR → MT → TTS (M2M100 多语言支持)")
parser.add_argument("--in-dir", default="data/processed/asr/commonvoice_demo/wav16k", help="输入音频目录")
parser.add_argument("--out-dir", default="outputs", help="输出音频目录")
parser.add_argument("--model", default="tiny", help="Whisper ASR 模型大小")
parser.add_argument("--target-lang", default="en", choices=list(TGT_LANG_MAP.keys()), help="目标语言")
parser.add_argument("--num-ex", type=int, default=5, help="处理样本数量")
parser.add_argument("--ref-trans", default="data/raw/asr/commonvoice_demo/transcripts.txt", help="参考转写")
args = parser.parse_args()

# ---------------------------
# 日志设置
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------------------
# 加载 Whisper 模型
# ---------------------------
logging.info(f"加载 Whisper 模型：{args.model}")
asr_model = whisper.load_model(args.model)

# ---------------------------
# 加载 M2M100 模型
# ---------------------------
MT_MODEL_NAME = "facebook/m2m100_418M"
logging.info(f"加载 M2M100 模型：{MT_MODEL_NAME}")
tokenizer = M2M100Tokenizer.from_pretrained(MT_MODEL_NAME)
mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_NAME)
tokenizer.src_lang = "zh"
tgt_lang = TGT_LANG_MAP[args.target_lang]

# ---------------------------
# 加载 TTS 模型
# ---------------------------
tts_model_name = LANG_TO_TTS_MODEL[args.target_lang]
logging.info(f"加载 TTS 模型：{tts_model_name}")
tts = TTS(model_name=tts_model_name, progress_bar=False, gpu=False)

# ---------------------------
# 目录准备
# ---------------------------
in_dir = args.in_dir
out_dir = os.path.join(args.out_dir, args.target_lang)
os.makedirs(out_dir, exist_ok=True)

ref_dict = {}
if os.path.exists(args.ref_trans):
    with open(args.ref_trans, encoding="utf-8") as f:
        for line in f:
            idx, txt = line.strip().split("|", 1)
            ref_dict[idx] = txt

ref_texts, hyp_texts = [], []

# ---------------------------
# 主处理流程
# ---------------------------
wav_files = sorted(f for f in os.listdir(in_dir) if f.endswith(".wav"))
total = min(args.num_ex, len(wav_files))

for i, fn in enumerate(tqdm(wav_files[:total], desc="Processing")):
    idx = fn.replace(".wav", "")
    wav_path = os.path.join(in_dir, fn)

    # ASR
    res = asr_model.transcribe(wav_path, language="zh")
    zh = res["text"].strip()

    # MT
    batch = tokenizer(zh, return_tensors="pt")
    gen = mt_model.generate(
        **batch,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    translated = tokenizer.decode(gen[0], skip_special_tokens=True)

    print(f"[{i+1}] {fn}")
    print("  ASR:", zh)
    print(f"   MT ({args.target_lang}):", translated)

    # TTS
    out_wav = os.path.join(out_dir, f"{idx}_{args.target_lang}.wav")
    tts.tts_to_file(text=translated, file_path=out_wav)
    print("  TTS →", out_wav)

    if idx in ref_dict:
        ref_texts.append(ref_dict[idx])
        hyp_texts.append(zh)

# ---------------------------
# 评估 WER（可选）
# ---------------------------
if ref_texts and hyp_texts:
    score = wer(ref_texts, hyp_texts)
    logging.info(f"WER (ASR only): {score:.2f}")

logging.info("✅ 完成！")
