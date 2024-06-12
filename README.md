# Jarvis - russian voice assistant

Demo video: [click here](https://youtu.be/fUJ_3pyTIcM)

## Key features

* Multi-language speech detection and recognition
* Multi-language speech synthesis with Viacheslav Baranov voice (the voice actor who voiced Jarvis in the Iron Man film series) 
* 8-bit quantized Mistral2 13B inside
* Low output latency due to outputs streaming. I've added queue and additional thread for audio playing (TODO: input STT streaming)
* Doesn't support short-term memory (I've tried LangChain and ended up fixing my ruined conda env for 2 hours straight)

## Models I have used:

* [Silero VAD model](https://github.com/snakers4/silero-vad) for speech detection
* [OpenAI Whisper](https://github.com/openai/whisper) Large-v3 with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition
* Saiga-Mistral 7B (russian Mistral) with [python bindings](https://github.com/abetlen/llama-cpp-python) for [llama.cpp](https://github.com/ggerganov/llama.cpp) with pretrained weights from [Ilya Gusev's HuggingFace](https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf)
* [Coqui XTTSv2](https://huggingface.co/coqui/XTTS-v2) for speech synthesis

## Usage

```bash
python jarvis.py
```

## Interaction example

```bash
(jarvis_env) peter@idk:~/PycharmProjects/jarvis$ python jarvis.py
Loading models...
Listening...
User:  Привет! Сколько весит склон?
Jarvis: Я не могу ответить на этот вопрос, так как не знаю, что такое "склон". Можешь уточнить, о чем идет речь?
Listening...
User:  Да, конечно, я имел ввиду не склон, а слон. Это такое животное.
Jarvis: Привет! Сколько весит слон? 

Слон - это крупное млекопитающее из семейства слоновых, обитающее в Африке и Южной Азии. Обычно слоны весят от 2
 до 7 тонн, но есть также экземпляры, которые могут достигать 10 тонн и более.
Listening...
User:  Хорошо. Какие виды слонов ты знаешь?
Jarvis: Существует два вида слонов - африканский слон (Loxodonta africana) и индийский слон (Elephas maximus). Африканские слоны имеют две формы рогов
, которые растут из их носа, в то время как у индийских слонов рога нет. Кроме того, африканские слоны обычно больше и тяжелее, чем индийские слоны.
Listening...
User:  Хороший ответ, спасибо. Напомни, пожалуйста, о чем был мой предыдущий вопрос.
Jarvis: Привет! Сколько весит склон? 
Склон - это неизвестный мне термин. Можешь уточнить, о чем идет речь?
Listening...
User:  Хорошо, и на твой встречный вопрос я ответил уточнением, и что же я тебе ответил?
Jarvis: Привет! Сколько весит слон?   Слон - это крупное млекопитающее из семейства слоновых, обитающее в Африке и Южной Азии. Обычно слоны весят от 2
 до 7 тонн, но есть также экземпляры, которые могут достигать 10 тонн и более.
```

## Installation

1. make conda environment with 
```bash
conda env create -f environment.yml
```
2. activate environment
```bash
conda activate jarvis_env
```
3. install llama-cpp-python (I've installed it with pip with custom index url, however conda doesn't record these)
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```
4. export env var
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib
```

## Computational resources

Large models require a lot of VRAM. Current version requires approximately 22-23GB of VRAM. 
This number might be drastically reduced by using models with lower number of params (Whisper-small, Mistral-7B, etc.). 


