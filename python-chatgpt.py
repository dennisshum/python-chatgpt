import os
import openai
import argparse

openai.api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser()
parser.add_argument("prompt", help="The prompt to send to the OpenAI API")
args = parser.parse_args()

completion = openai.Completion.create(
  model="text-davinci-003",
  prompt=f"Write Python script to {args.prompt}",
  max_tokens=500,
  temperature=0.5
)

print(completion.choices[0].text)