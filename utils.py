import time
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from config import config
import sounddevice as sd
from threading import Thread, Lock
from queue import Queue


@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out


def start_audio_output_worker():
    def audio_output_worker():
        while True:
            if audio_output_queue.empty():
                time.sleep(0.1)
                continue
            audio_output_lock.acquire()
            tts_output_item = audio_output_queue.get()
            if tts_output_item == 'exit':
                audio_output_queue.task_done()
                return
            sd.play(tts_output_item, config['output_audio_sampling_rate'])
            sd.wait()
            audio_output_queue.task_done()
            audio_output_lock.release()

    audio_output_queue = Queue()
    audio_output_lock = Lock()
    audio_output_thread = Thread(target=audio_output_worker)
    audio_output_thread.start()
    return audio_output_queue, audio_output_lock, audio_output_thread


def finish_audio_output_worker(audio_output_queue, audio_output_thread):
    audio_output_queue.put('exit')
    audio_output_queue.join()
    audio_output_thread.join()
