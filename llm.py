from openai import OpenAI
from openai.types.chat import ChatCompletion
import numpy as np
import os
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL")

client = OpenAI(
    api_key=API_KEY, base_url=BASE_URL
)


def get_completion(system_prompt: str, user_message: str) -> ChatCompletion:
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        model="gpt-4-turbo",
        top_logprobs=4,
        logprobs=True,
    )
    return completion


def get_yes_no_probs(completion: ChatCompletion) -> tuple[float, float]:
    top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
    try: 
        yes_logprob = list(filter(lambda x: x.token.lower() == "ja", top_logprobs))[
            0
        ].logprob
    except IndexError:
        yes_logprob = - np.inf
    try: 
        no_logprob = list(filter(lambda x: x.token.lower() == "ne", top_logprobs))[
            0
        ].logprob
    except IndexError:
        no_logprob = - np.inf

    yes_prob = np.exp(yes_logprob)
    no_prob = np.exp(no_logprob)

    yes_prob = yes_prob / (yes_prob + no_prob)
    no_prob = no_prob / (yes_prob + no_prob)

    return yes_prob, no_prob
