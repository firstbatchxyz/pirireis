import anthropic
import os
from openai import OpenAI
from oaib import Auto, Batch
from decouple import config
from enum import Enum

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

class ModelEnum(str, Enum):
    """Enum for OpenAI models"""
    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_4_0125_preview = "gpt-4-0125-preview"

class OpenAIWorker:
    def __init__(self):
        self.client = OpenAI()
        self.batch = Batch(rpm=100, tpm=1_000, workers=5)
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def hard_reset(self):
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def ask(self, sys_prompt, prompt, model=ModelEnum.gpt_3_5_turbo):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    async def ask_async(self, sys_prompt, prompts):
        for prompt in prompts:
            await self.batch.add(
                "chat.completions.create",
                model=ModelEnum.gpt_3_5_turbo,
                messages= [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
            )
        results = await self.batch.run()
        results = results.get("result")
        graphs = [response["choices"][0]["message"]["content"] for response in results]
        return graphs

class Haiku:
    def __init__(self):
        self.client = anthropic.Client(api_key=config("ANTHROPIC_API_KEY"))

    def generate(self, SYS_PROMPT, prompt):
        if SYS_PROMPT is None:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}  # <-- user prompt
                ]
            )
        else:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                system=SYS_PROMPT,  # <-- system prompt
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}  # <-- user prompt
                ]
            )
        return response.content[0].text

class YiChat:
    def __init__(self):
        self.openai = OpenAI(
            api_key=config("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def __call__(self, text: str, system_prompt: str):
        resp = self.openai.chat.completions.create(
            model="01-ai/Yi-34B-Chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
        )
        return resp.choices[0].message.content

if __name__ == "__main__":
    print(ModelEnum.gpt_3_5_turbo == "gpt-3.5-turbo")
    mixtral = YiChat()
    print(mixtral("What is your name (of the model)?", ""))

    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
        api_key=config("DEEPINFRA_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai",
    )

    chat_completion = openai.chat.completions.create(
        model="01-ai/Yi-34B-Chat",
        messages=[{"role": "user", "content": "What is your name (of the model)?"}],
    )

    print(chat_completion.choices[0].message.content)