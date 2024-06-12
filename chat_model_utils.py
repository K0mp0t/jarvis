def get_message_tokens(model, role, content):
    content = f"{role}\n{content}\n</s>"
    content = content.encode("utf-8")
    message_tokens = model.tokenize(content, special=True)
    return message_tokens


def get_system_tokens(model, system_prompt):
    system_message = {
        "role": "system",
        "content": system_prompt
    }
    return get_message_tokens(model, **system_message)


def generate_chat_model_output(model, message, config):
    message_tokens = get_message_tokens(model=model, role="user", content=message)
    role_tokens = model.tokenize("bot\n".encode("utf-8"), special=True)
    tokens = get_system_tokens(model, config['chat_model_system_prompt'])
    tokens += message_tokens + role_tokens
    generator = model.generate(
        tokens,
        top_k=config['top_k'],
        top_p=config['top_p'],
        temp=config['temperature'],
        repeat_penalty=config['repeat_penalty']
    )
    for i, token in enumerate(generator):
        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos() or len(tokens) > config['chat_model_n_ctx'] - 1:
            break

        yield token_str


class MemoryModule(object):
    def __init__(self, memory_size: int):
        self.inputs = list()
        self.outputs = list()
        self.memory_size = memory_size

    def process_input(self, input: str) -> str:
        if len(self.inputs) >= self.memory_size or len(self.outputs) >= self.memory_size:
            self.inputs.pop(0)
            self.outputs.pop(0)

        self.inputs.append(input)

        assert len(self.inputs) == len(self.outputs) + 1

        return ('\n'.join('Пользователь: ' + inp + '\nДжарвис: ' + out for inp, out in zip(self.inputs, self.outputs)) +
                '\nПользователь:' + input + '\nДжарвис:')

    def process_output(self, output: str) -> None:
        if len(self.inputs) >= self.memory_size or len(self.outputs) >= self.memory_size:
            self.inputs.pop(0)
            self.outputs.pop(0)

        self.outputs.append(output)
