import os.path
from typing import Dict, Any

from llama_cpp import Llama
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model
from TTS.api import TTS

from chat_model_utils import get_system_tokens
from utils import suppress_stdout_stderr


def load_models(config: Dict[str, Any]):
    print('Loading models...')

    vad_model = get_vad_model()
    stt_model = WhisperModel(config['whisper_model_size'], download_root="./models", device=config['device'],
                                    compute_type="float16")

    chat_model = Llama(
        model_path=config['chat_model_path'],
        n_ctx=config['chat_model_n_ctx'],
        n_parts=1,
        n_gpu_layers=-1,
        verbose=False
    )

    system_tokens = get_system_tokens(chat_model, config['chat_model_system_prompt'])
    tokens = system_tokens
    chat_model.eval(tokens)

    # chat_model = prepare_langchain_pipeline(config)

    with suppress_stdout_stderr():
        tts_model = TTS(model_path=config['tts_model_path'],
                        config_path=os.path.join(config['tts_model_path'], 'config.json')).to(config['device'])

    return vad_model, stt_model, chat_model, tts_model
