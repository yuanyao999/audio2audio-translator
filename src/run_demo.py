import os
import argparse
import logging
import whisper
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from jiwer import wer
from TTS.api import TTS  # Coqui TTS

# ---------------------------
# Argument Parsing
# ---------------------------
parser = argparse.ArgumentParser(description="Demo script: ASR→MT→TTS using Coqui TTS only")
parser.add_argument(
    "--in-dir",
    default="data/processed/asr/commonvoice_demo/wav16k",
    help="Input directory with 16 kHz WAV files",
)
parser.add_argument(
    "--out-dir",
    default="outputs",
    help="Output directory for synthesized WAV files",
)
parser.add_argument(
    "--model",
    default="tiny",
    help="Whisper ASR model size (tiny/base/small/medium/large)",
)
parser.add_argument(
    "--num-ex",
    type=int,
    default=5,
    help="Number of examples to process",
)
parser.add_argument(
    "--ref-trans",
    default="data/raw/asr/commonvoice_demo/transcripts.txt",
    help="Reference transcripts for WER evaluation",
)
args = parser.parse_args()

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# ---------------------------
# Load Models
# ---------------------------
logging.info(f"Loading Whisper model: {args.model}")
asr_model = whisper.load_model(args.model)

mt_name = "Helsinki-NLP/opus-mt-zh-en"
logging.info(f"Loading MT model: {mt_name}")
tokenizer = MarianTokenizer.from_pretrained(mt_name)
mt_model  = MarianMTModel.from_pretrained(mt_name)

# ---------------------------
# Initialize Coqui TTS
# ---------------------------
logging.info("Initializing Coqui TTS")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# ---------------------------
# Prepare I/O
# ---------------------------
in_dir  = args.in_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# Load reference transcripts
ref_dict = {}
if os.path.exists(args.ref_trans):
    with open(args.ref_trans, encoding="utf-8") as f:
        for line in f:
            idx, txt = line.strip().split("|", 1)
            ref_dict[idx] = txt

# Metric lists
hyp_texts = []
ref_texts = []

# ---------------------------
# Run Demo
# ---------------------------
wav_files = sorted(f for f in os.listdir(in_dir) if f.endswith(".wav"))
total     = min(args.num_ex, len(wav_files))

for i, fn in enumerate(tqdm(wav_files[:total], desc="Processing")):
    idx      = fn.replace(".wav", "")
    wav_path = os.path.join(in_dir, fn)

    # ASR → 中文
    res = asr_model.transcribe(wav_path)
    zh  = res["text"].strip()

    # MT → 英文
    batch = tokenizer(zh, return_tensors="pt", padding=True)
    gen   = mt_model.generate(**batch)
    en    = tokenizer.decode(gen[0], skip_special_tokens=True)

    print(f"[{i+1}] {fn}")
    print("  ASR:", zh)
    print("   MT:", en)

    # TTS → 英语 WAV (Coqui)
    out_wav = os.path.join(out_dir, f"{idx}_en.wav")
    tts.tts_to_file(text=en, file_path=out_wav)
    print("  TTS →", out_wav)

    # Collect for WER metric
    if idx in ref_dict:
        ref_texts.append(ref_dict[idx])
        hyp_texts.append(zh)

# ---------------------------
# Evaluate WER
# ---------------------------
if ref_texts and hyp_texts:
    logging.info(f"WER: {wer(ref_texts, hyp_texts):.2f}")

logging.info("Demo complete.")
