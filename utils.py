from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from config import config
import sounddevice as sd
from threading import Thread
from queue import Queue
import time


@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out


def start_audio_output_worker():
    def audio_output_worker():
        while True:
            tts_output_item = audio_output_queue.get()
            if tts_output_item == 'exit':
                audio_output_queue.task_done()
                return
            sd.play(tts_output_item, config['output_audio_sampling_rate'])
            # sd.wait()  doesn't work for some reason
            # so I have to estimate the length of the audio as sleep while it is playing
            print(len(tts_output_item) / config['output_audio_sampling_rate'])
            time.sleep(len(tts_output_item) / config['output_audio_sampling_rate'])
            audio_output_queue.task_done()

    audio_output_queue = Queue()
    audio_output_thread = Thread(target=audio_output_worker)
    audio_output_thread.start()
    return audio_output_queue, audio_output_thread


def finish_audio_output_worker(audio_output_queue, audio_output_thread):
    audio_output_queue.put('exit')
    audio_output_queue.join()
    audio_output_thread.join()
