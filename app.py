import os
import tempfile
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import gradio as gr

# Load models once
asr_model = whisper.load_model("tiny")
mt_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(mt_name)
mt_model = MarianMTModel.from_pretrained(mt_name)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Pipeline function
def translate_audio(audio):
    # audio: tuple (sample_rate, numpy array) or file path
    if isinstance(audio, tuple):
        sr, data = audio
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        import soundfile as sf
        sf.write(tmp.name, data, sr)
        wav_path = tmp.name
    else:
        wav_path = audio

    # ASR
    res = asr_model.transcribe(wav_path)
    zh = res["text"].strip()

    # MT
    batch = tokenizer(zh, return_tensors="pt", padding=True)
    gen = mt_model.generate(**batch)
    en = tokenizer.decode(gen[0], skip_special_tokens=True)

    # TTS
    out_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    tts.tts_to_file(text=en, file_path=out_wav)

    return zh, en, out_wav

# Gradio UI definition
with gr.Blocks() as demo:
    gr.Markdown("# 音频到音频同声翻译 Demo")
    with gr.Row():
        audio_in = gr.Audio(sources=["upload"], type="filepath", label="上传中文语音 (WAV)")
        with gr.Column():
            txt_zh = gr.Textbox(label="ASR 输出 (中文)")
            txt_en = gr.Textbox(label="MT 输出 (英文)")
            audio_out = gr.Audio(label="TTS 输出 (英语)")
    btn = gr.Button("翻译并合成")
    btn.click(fn=translate_audio, inputs=audio_in, outputs=[txt_zh, txt_en, audio_out])

if __name__ == "__main__":
    demo.launch(share=True)  
