import numpy as np
import pyaudio
import time

from config import config
from models_loader import load_models
from chat_model_utils import process_input
from utils import suppress_stdout_stderr, start_audio_output_worker, finish_audio_output_worker
from vad_utils import detect_voice_activity

vad_model, stt_model, chat_model, tts_model = load_models(config)

p = pyaudio.PyAudio()

stream = p.open(format=config['dtype'],
                channels=config['channels'],
                rate=config['input_audio_sampling_rate'],
                input=True,
                frames_per_buffer=config['chunk'])

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

    # if len(audio_data) < config['min_voice_input_duration_s'] * config['input_audio_sampling_rate'] / config['chunk']:
    #     print('No voice detected. Repeating...')
    #     if audio_output_lock.locked():
    #         audio_output_lock.release()
    #     continue

    audio_data = np.hstack(audio_data)

    segments, info = stt_model.transcribe(audio_data, language='ru', beam_size=5)

    llm_input = " ".join([segment.text for segment in segments])

    print('User:', llm_input)

    llm_output_generator = process_input(chat_model, llm_input, config)

    llm_output = ""
    print('Jarvis: ', end="")
    number_of_chars_printed = len('Jarvis: ')

    audio_output_lock.release()

    for out_token in llm_output_generator:
        llm_output += out_token
        number_of_chars_printed += len(out_token)
        if number_of_chars_printed > config['max_print_line_length']:
            number_of_chars_printed = len(out_token)
            out_token = "\n" + out_token
        print(out_token, end="", flush=True)
        if any(out_token.endswith(punct) for punct in ['.', '?', '!', '...']) and len(llm_output) > 10:
            llm_output = llm_output.replace('\n', ' ')
            with suppress_stdout_stderr():
                tts_output = tts_model.tts(llm_output, speaker_wav=config['tts_model_speaker_wav_fp'],
                                           language='ru', split_sentences=False)
            audio_output_queue.put(tts_output)
            llm_output = ""

    print()

finish_audio_output_worker(audio_output_queue, audio_output_thread)
