# Jarvis - russian voice assistant

## Key features

* Multi-language speech detection and recognition
* Multi-language speech synthesis with Viacheslav Baranov voice (the voice actor who voiced Jarvis in the Iron Man film series) 
* 8-bit quantized Mistral 7B inside

## Models I have used:

* [Silero VAD model](https://github.com/snakers4/silero-vad) for speech detection
* [OpenAI Whisper](https://github.com/openai/whisper) Large-v3 with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition
* Saiga-Mistral 7B (russian Mistral) with [python bindings](https://github.com/abetlen/llama-cpp-python) for [llama.cpp](https://github.com/ggerganov/llama.cpp) with pretrained weights from [Ilya Gusev HuggingFace](https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf)
* [Coqui XTTSv2](https://huggingface.co/coqui/XTTS-v2) for speech synthesis

## Usage

```bash
python jarvis.py
```

## Interaction example

```commandline
(jarvis_env) peter@idk:~/PycharmProjects/jarvis$ python jarvis.py
Loading models...
Listening...
User:  Привет! Расскажи, пожалуйста, какие бывают слоны и где они обитают?
Jarvis:  Приветствую! Слоны - это крупные млекопитающие из семейства слоновых (Elephantidae). Они делятся на два вида: африканский слон (Loxodonta africana) и индийский слон (Elephas maximus).

Африканские слоны обитают в саваннах, лесах и других открытых пространствах на большей части континента. Они имеют более крупные уши и более длинный рог, чем индийские слоны.

Индийские слоны обитают в тропических лесах Юго-Восточной Азии, включая страны такие как Индия, Непал, Шри-Ланка и Мьянма. Они имеют более короткие уши и рог, чем африканские слоны.

Оба вида слонов являются крупными травоядными животными, которые питаются различными растениями, такими как трава, листья и фрукты.
Listening...
No voice detected. Repeating...
Listening...
User:  Отличный ответ. Спасибо.  Расскажи, пожалуйста, о видах львов.
Jarvis: Вы хотите узнать о разных видах львов? Хорошо!

Львы - это крупные кошачьи, которые обитают в Африке и на Ближнем Востоке. Они принадлежат к роду Panthera, который также включает тигров, леопарда и ягуара.

Выделяют несколько видов львов:

1. Африканский (или восточно-африканский) лв - это наиболее распространенный вид, который обитает в саваннах и лесах Восточной и Южной Африки.
2. Индийский (или западно-африканский) лв - это меньшее по размеру животное, которое обитает в Западной и Центральной Африке.
3. Южно-американский лв - это вымерший вид, который жил на территории современных Бразилии и Перу.

Кроме того, существуют различные подвиды львов, которые имеют некоторые отличия в окраске шерсти и размерах тела.

```