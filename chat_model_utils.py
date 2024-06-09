# from langchain_community.llms import LlamaCpp
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
# from langchain_core.prompts import PromptTemplate


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


def process_input(model, message, config):
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


# def prepare_langchain_pipeline(config):
#     template = """User: {question}
#     Jarvis: Сэр, я отвечу на любой вопрос."""
#
#     prompt = PromptTemplate.from_template(template)
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#
#     chat_model = LlamaCpp(
#         model_path=config['chat_model_path'],
#         n_ctx=config['chat_model_n_ctx'],
#         n_parts=1,
#         n_gpu_layers=-1,
#         verbose=False,
#         callback_manager=callback_manager,
#     )
#
#     system_tokens = get_system_tokens(chat_model, config['chat_model_system_prompt'])
#     tokens = system_tokens
#     chat_model.eval(tokens)
#
#     llm_chain = prompt | chat_model
#
#     return llm_chain
