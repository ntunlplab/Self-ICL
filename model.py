import os
import time
import openai
from prompt import StreamPrompt, BatchPrompt
from colorama import Fore, Style
from argparse import Namespace

class Model(object):
    api_interval = 1.0 # seconds
    cost_per_1000tokens = 0.02

    def __init__(self, config: Namespace):
        self._config = config
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def retry_with_exponential_backoff(
        func,
        initial_delay: float = 5,
        exponential_base: float = 2,
        max_retries: int = 5,
        errors: tuple = (openai.error.RateLimitError,),
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
    def complete(self, prompt: str) -> dict:
        time.sleep(Model.api_interval)
        return openai.Completion.create(prompt=prompt, **vars(self._config))
    
    def set_api_key(self, key: str) -> None:
        openai.api_key = key

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
    
    # prompt
    tasks = {
        "boolean_expressions": {
            "task_desc": "Evaluate the result of a random Boolean expression.",
            "inputs": "not ( True ) and ( True ) is"
        },
        "causal_judgement": {
            "task_desc": "Answer questions about causal attribution.",
            "inputs": "How would a typical person answer each of the following questions about causation?\nA machine is set up in such a way that it will short circuit if both the black wire and the red wire touch the battery at the same time. The machine will not short circuit if just one of these wires touches the battery. The black wire is designated as the one that is supposed to touch the battery, while the red wire is supposed to remain in some other part of the machine. One day, the black wire and the red wire both end up touching the battery at the same time. There is a short circuit. Did the black wire cause the short circuit?\nOptions:\n- Yes\n- No"
        },
        "date_understanding": {
            "task_desc": "Infer the date from context.",
            "inputs": "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937\n(C) 01/04/1938\n(D) 12/04/1937\n(E) 12/25/2006\n(F) 07/25/1937"
        }
    }
    num_demos = 3
    zero_shots = []

    task_name = "date_understanding"
    stream_prompt = StreamPrompt(
        task_desc=tasks[task_name]["task_desc"],
        inputs=tasks[task_name]["inputs"],
        num_demos=num_demos,
        shots=zero_shots
    )

    prompt = stream_prompt.gen_demo_inputs()
    res_text = model_api.complete(prompt)["choices"][0]["text"]
    full_text = prompt + res_text
    print(full_text)