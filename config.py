import pyaudio


config = dict(
    # paths
    whisper_model_size='large-v3',
    chat_model_path='./models/saiga2-13b-q8_0.gguf',
    tts_model_path='./models/tts_models--multilingual--multi-dataset--xtts_v2',

    # global settings
    device='cuda',
    max_print_line_length=150,

    # audio input settings
    chunk=1024,
    dtype=pyaudio.paInt16,
    channels=1,
    input_audio_sampling_rate=16000,
    output_audio_sampling_rate=24000,
    max_record_seconds=15,

    # vad model settings
    vad_threshold=0.5,
    max_silence_duration_s=2,
    min_voice_input_duration_s=1,

    # chat model settings
    chat_model_system_prompt="""Ты — Джарвис, русскоязычный автоматический ассистент. Ты разговариваешь с людьми 
                                и помогаешь им. Если у тебя нет ответа на заданный тебе вопрос, отвечай, что не знаешь
                                или проси уточнить вопрос. У тебя нет доступа к интернету""",
    chat_model_n_ctx=2000,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repeat_penalty=1.1,
    chat_model_memory_size=5,

    # tts model settings
    tts_model_speaker_wav_fp='./jarvis_sample1.wav'
)
