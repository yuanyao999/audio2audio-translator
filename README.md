# Audio-to-Audio Language Translator

本项目实现了一个端到端的**音频到音频**同声翻译器，能够将普通话语音（Putonghua）翻译并合成成英语语音。流水线包括：

1. **ASR （自动语音识别）**：使用 OpenAI Whisper
2. **MT （机器翻译）**：使用 Helsinki-NLP MarianMT
3. **TTS （文本转语音）**：使用 Coqui TTS

---

## 项目结构

```
audio2audio-translator/
├── application/                     # 部署脚本与模型下载说明（可选）
│   └── run.py                       # 部署入口示例（Flask/FastAPI）
├── data/                            # 数据目录
│   ├── raw/asr/commonvoice_demo/    # Common Voice 子集原始音频与转写
│   └── processed/asr/commonvoice_demo/  # 重采样后音频与清洗转写
├── outputs/                         # 生成的英语语音文件
├── src/
│   ├── utils/
│   │   └── extract_commonvoice_demo.py  # 提取 Common Voice 子集脚本
│   └── run_demo.py                   # ASR→MT→TTS 演示脚本（Coqui TTS 版）
├── venv/                            # Python 虚拟环境（未上传）
├── requirements.txt                 # Python 依赖清单
└── README.md                        # 本说明文件
└── app.py 
```

---

## 环境与依赖

* **Python**：建议使用 3.10.x
* 创建并激活虚拟环境：

  ```bash
  python3.10 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  ```
* 安装依赖：

  ```bash
  pip install \
    openai-whisper \
    torch torchaudio \
    transformers sentencepiece \
    datasets soundfile librosa \
    tqdm jiwer sacrebleu \
    TTS
  ```

---

## 数据准备

1. 提取 Common Voice 中文子集（约 30 分钟）：

   ```bash
   python src/utils/extract_commonvoice_demo.py
   ```

2. 重采样并清洗：

**重采样到 16kHz**

```bash
mkdir -p data/processed/asr/commonvoice_demo/wav16k
for f in data/raw/asr/commonvoice_demo/wav/*.wav; do
  python - << 'PYCODE'
import librosa, soundfile as sf, sys, os
inp=sys.argv[1]
out=inp.replace('raw/asr/commonvoice_demo/wav/', 'processed/asr/commonvoice_demo/wav16k/')
y,_ = librosa.load(inp, sr=16000)
os.makedirs(os.path.dirname(out), exist_ok=True)
sf.write(out, y, 16000)
print('Resampled →', out)
PYCODE
  "$f"
done
```

**文本清洗**

```bash
python - << 'PYCODE'
import os
fin=open('data/raw/asr/commonvoice_demo/transcripts.txt','r',encoding='utf-8')
fout=open('data/processed/asr/commonvoice_demo/transcripts_clean.txt','w',encoding='utf-8')
for line in fin:
    idx, txt = line.strip().split('|',1)
    fout.write(f"{idx}|{txt.strip().lower()}
")
fin.close(); fout.close()
print('文本清洗完成')
PYCODE
```

---

## 演示脚本

主要脚本：`src/run_demo.py`，功能：

* 加载 Whisper ASR 模型，识别中文语音
* 加载 MarianMT 模型，翻译为英文文本
* 使用 Coqui TTS 合成英语语音（WAV）
* 计算并输出 WER（可与参考转写对比）

示例运行：

```bash
python src/run_demo.py --in-dir data/processed/asr/commonvoice_demo/wav16k \
                       --out-dir outputs \
                       --model tiny \
                       --num-ex 5 \
                       --ref-trans data/raw/asr/commonvoice_demo/transcripts.txt
```

---

## 结果

* 脚本将打印每条示例的中文转写（ASR）和英文翻译（MT）
* 在 `outputs/` 目录下生成相应的英语 WAV 音频文件
* 在日志中输出 WER 评分

---
## 🖥️ 可视化界面

本项目提供一个 Gradio 网页界面：

```bash
pip install gradio
python app.py


## 模型说明

项目使用的所有模型均为**预训练模型**：

* Whisper ASR：`tiny` 版本，从 OpenAI 官方自动下载
* MarianMT 翻译：Helsinki-NLP/opus-mt-zh-en，从 Hugging Face Hub 下载
* Coqui TTS：使用 `tts_models/en/ljspeech/tacotron2-DDC` 模型

如需替换或定制自己的微调模型，请参考 **application/models/** 下的下载或导出脚本。

---

## 许可证和引用

* 数据集：Common Voice — Mozilla Public License v2.0
* ASR 模型：OpenAI Whisper
* MT 模型：Helsinki-NLP MarianMT
* TTS 模型：Coqui TTS

---

*作者：Yuanyao ZUO*
