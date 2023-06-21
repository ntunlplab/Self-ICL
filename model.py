import os
import time
import openai
import tiktoken
from typing import Set, Tuple
from prompt import StreamPrompt, BatchPrompt
from colorama import Fore, Style
from argparse import Namespace

class Model(object):
    api_interval = 0.05 # seconds
    cost_per_1000tokens = 0.02

    def __init__(self, config: Namespace):
        self._config = config
        self._tokenizer = tiktoken.encoding_for_model(self._config.model)
        self._original_max_tokens = self._config.max_tokens
        self._original_temperature = self._config.temperature
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def retry_with_exponential_backoff(
        func,
        initial_delay: float = 0.5,
        exponential_base: float = 2,
        max_retries: int = 20,
        errors: tuple = (openai.error.RateLimitError, openai.error.APIError),
    ):
        """Retry a function with exponential backoff."""
    
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay
            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                # Retry on specific errors
                except errors as e:
                    # Increment retries
                    num_retries += 1
                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            Fore.RED + f"Maximum number of retries ({max_retries}) exceeded." + Style.RESET_ALL
                        )
                    # Increment the delay
                    delay *= exponential_base
                    # Sleep for the delay
                    print(Fore.YELLOW + f"Error encountered. Retry ({num_retries}) after {delay} seconds..." + Style.RESET_ALL)
                    time.sleep(delay)
                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e
    
        return wrapper

    @retry_with_exponential_backoff
    def complete(self, prompt: str, label_set: Set[str] = set(), temperature: float = None) -> Tuple[str, str]:
        time.sleep(Model.api_interval)
        params = vars(self._config)
        if label_set:
            max_tokens = 0
            for label in label_set:
                tokens = label
                if '(' == tokens[0]:
                    tokens = tokens[1:] # remove parentheses
                    if prompt[-1] != '(':
                        prompt += " ("
                token_ids = self._tokenizer.encode(tokens)
                max_tokens = max(max_tokens, len(token_ids))
            if max_tokens > 0:
                # params["logit_bias"] = logit_bias
                if prompt[-1] != '(':
                    max_tokens += 1 # +1 for whitespace
                params["max_tokens"] = max_tokens
        else:
            params["max_tokens"] = self._original_max_tokens
        
        if temperature is not None:
            params["temperature"] = temperature
        else:
            params["temperature"] = self._original_temperature
        print(Fore.GREEN + f" (Predicting with temperature: {params['temperature']}, max_tokens: {params['max_tokens']}) " + Style.RESET_ALL, end='')
        return prompt, openai.Completion.create(prompt=prompt, **params)["choices"][0]["text"]
    
    def set_api_key(self, key: str) -> None:
        openai.api_key = key

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

# for unit testing
if __name__ == "__main__":
    # model
    config = Namespace(
        model="text-davinci-003",
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
    )
    model_api = Model(config)
    
    # tasks
    from task import TaskGenerator
    
    task_gen = TaskGenerator(
        task_input_path="./bbh/BIG-Bench-Hard/bbh/",
        task_desc_path="./bbh/bbh_task_description.json",
        batch_size=1
    )
    task_names = ["boolean_expressions", "causal_judgement", "date_understanding", "formal_fallacies", "sports_understanding"]
    
    # prompt
    for task_name in task_names:
        task = task_gen.get_task(task_name)
        task_inputs = task.get_new_inputs()
        stream_prompt = StreamPrompt(
            task_desc=task.task_desc,
            inputs=task_inputs,
            num_demos=3,
            shots=[]
        )
        
        pred_prompt = stream_prompt.gen_prediction()
        print(f"Generating prediction from label set: {task.label_set} ->")
        pred_prompt, res_text = model_api.complete(pred_prompt, task.label_set)
        full_text = pred_prompt + res_text
        print(f"full text: {full_text}")
