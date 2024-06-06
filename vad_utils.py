def detect_voice_activity(audio_data_chunk, config, model, model_state=None):
    if model_state is None:
        model_state = model.get_initial_state(batch_size=1)

    speech_prob, model_state = model(audio_data_chunk, model_state, config['input_audio_sampling_rate'])

    return speech_prob > config['vad_threshold'], model_state
