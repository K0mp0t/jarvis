import numpy as np
import pyaudio
import time

from config import config
from models_loader import load_models
from chat_model_utils import generate_chat_model_output, MemoryModule
from utils import suppress_stdout_stderr, start_audio_output_worker, finish_audio_output_worker
from vad_utils import detect_voice_activity

stream = pyaudio.PyAudio().open(format=config['dtype'],
                                channels=config['channels'],
                                rate=config['input_audio_sampling_rate'],
                                input=True,
                                frames_per_buffer=config['chunk'])

vad_model, stt_model, chat_model, tts_model = load_models(config)
chat_model_memory = MemoryModule(memory_size=config['chat_model_memory_size'])

vad_model_state = None
audio_output_queue, audio_output_lock, audio_output_thread = start_audio_output_worker()

while True:
    if not audio_output_lock.locked():
        audio_output_lock.acquire()
    else:
        time.sleep(0.01)
        continue

    print('Listening...')
    audio_data = list()
    last_positive_chunk_idx = 0
    chunks = 0

    while True:
        chunks += 1
        data = stream.read(config['chunk'])
        data = np.frombuffer(data, dtype=np.int16)

        speech_detected, vad_model_state = detect_voice_activity(
            data.astype('float32') / config['input_audio_sampling_rate'],
            config, vad_model, vad_model_state)
        if speech_detected:
            audio_data.append(data)
            last_positive_chunk_idx = chunks

        if ((chunks - last_positive_chunk_idx > config['max_silence_duration_s'] *
            config['input_audio_sampling_rate'] / config['chunk']) and len(audio_data) >
                config['min_voice_input_duration_s'] * config['input_audio_sampling_rate'] / config['chunk']):
            break

    segments, info = stt_model.transcribe(np.hstack(audio_data), language='ru', beam_size=5)
    chat_model_input = " ".join([segment.text for segment in segments])
    print('User:', chat_model_input)

    chat_model_input = chat_model_memory.process_input(chat_model_input)
    chat_model_output_generator = generate_chat_model_output(chat_model, chat_model_input, config)

    chat_model_output = ""
    chat_model_output_tss_index = 0
    print('Jarvis: ', end="")
    number_of_chars_printed = len('Jarvis: ')

    audio_output_lock.release()

    for out_token in chat_model_output_generator:
        chat_model_output += out_token
        number_of_chars_printed += len(out_token)
        if number_of_chars_printed > config['max_print_line_length']:
            number_of_chars_printed = len(out_token)
            out_token = "\n" + out_token
        print(out_token, end="", flush=True)
        if any(out_token.endswith(punct) for punct in ['.', '?', '!', '...']) and len(chat_model_output) > 10:
            chat_model_output = chat_model_output.replace('\n', ' ')
            with suppress_stdout_stderr():
                tts_output = tts_model.tts(chat_model_output[chat_model_output_tss_index:],
                                           speaker_wav=config['tts_model_speaker_wav_fp'],
                                           language='ru', split_sentences=False)
            audio_output_queue.put(tts_output)
            chat_model_output_tss_index = len(chat_model_output)

    chat_model_memory.process_output(chat_model_output)

    print()

finish_audio_output_worker(audio_output_queue, audio_output_thread)
