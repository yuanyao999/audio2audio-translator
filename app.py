import os
import tempfile
import whisper
import gradio as gr
import soundfile as sf
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from TTS.api import TTS

# ---------------------------
# 支持语言映射（已去除西班牙语）
# ---------------------------
TGT_LANG_MAP = {
    "英语（en）": "en",
    "法语（fr）": "fr",
    "德语（de）": "de",
}

TTS_MODELS = {
    "en": "tts_models/en/ljspeech/tacotron2-DDC",
    "fr": "tts_models/fr/css10/vits",
    "de": "tts_models/de/thorsten/tacotron2-DCA",
}

# ---------------------------
# 加载 Whisper ASR 模型
# ---------------------------
print("Loading Whisper ASR...")
asr_model = whisper.load_model("tiny")

# ---------------------------
# 加载 M2M100 翻译模型
# ---------------------------
print("Loading M2M100 translation model...")
MT_MODEL_NAME = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(MT_MODEL_NAME)
mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_MODEL_NAME)

# ---------------------------
# 翻译主函数（支持多语言）
# ---------------------------
def translate_audio(audio, lang_display):
    tgt_lang = TGT_LANG_MAP[lang_display]

    # 保存上传的音频
    if isinstance(audio, tuple):
        sr, data = audio
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, data, sr)
        wav_path = tmp.name
    else:
        wav_path = audio

    # ASR：中文识别
    res = asr_model.transcribe(wav_path, language="zh")
    zh = res["text"].strip()

    # MT：中文 → 目标语言
    tokenizer.src_lang = "zh"
    inputs = tokenizer(zh, return_tensors="pt")
    gen = mt_model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    translated = tokenizer.decode(gen[0], skip_special_tokens=True)

    # TTS：目标语言语音合成
    tts_model = TTS(model_name=TTS_MODELS[tgt_lang], progress_bar=False, gpu=False)
    out_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    tts_model.tts_to_file(text=translated, file_path=out_wav)

    return zh, translated, out_wav

# ---------------------------
# Gradio 网页界面
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ 中文语音 → 多语言语音翻译演示")

    with gr.Row():
        audio_in = gr.Audio(sources=["upload"], type="filepath", label="上传中文语音 (WAV)")
        lang_dropdown = gr.Dropdown(label="选择输出语言", choices=list(TGT_LANG_MAP.keys()), value="英语（en）")

    with gr.Row():
        txt_zh = gr.Textbox(label="ASR 输出（中文）")
        txt_mt = gr.Textbox(label="MT 翻译输出")
        audio_out = gr.Audio(label="TTS 合成语音")

    btn = gr.Button("翻译并合成")
    btn.click(fn=translate_audio, inputs=[audio_in, lang_dropdown], outputs=[txt_zh, txt_mt, audio_out])

if __name__ == "__main__":
    demo.launch(share=True)
