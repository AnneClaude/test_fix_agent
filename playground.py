import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads your .env file

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=60,
)

# --- The real tool: only YOUR code runs this ---

def add(a, b):
    return a + b

# --- The tool's description: this is what the MODEL reads ---

tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]

# --- Step 1: ask the question, and tell the model which tools exist ---

messages = [{"role": "user", "content": "What is 17 + 25? Use the add tool."}]

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=messages,
    tools=tools,
)

# --- Step 2: the model replies with a tool request instead of an answer ---

message = response.choices[0].message
print("finish_reason:", response.choices[0].finish_reason)
print("tool_calls:", message.tool_calls)

# --- Step 3: YOUR code does the work ---

call = message.tool_calls[0]                  # the first (and only) tool request
args = json.loads(call.function.arguments)    # '{"b":25,"a":17}' (text) -> {"b": 25, "a": 17} (dict)
result = add(**args)                          # same as add(a=17, b=25)
print("I ran", call.function.name, "with", args, "->", result)

# --- Step 4: report back by adding to the conversation history ---

messages.append(message)                      # the model's request (keeps its id + thought signature)
messages.append({
    "role": "tool",                           # this message is a tool result, not the user talking
    "tool_call_id": call.id,                  # which request this answers
    "content": str(result),                   # content must be text
})

# --- Step 5: call the model again with the full history ---

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=messages,
    tools=tools,
)

print("finish_reason:", response.choices[0].finish_reason)
print("answer:", response.choices[0].message.content)