---
title: "Project: AI Evals on a Simple Chatbot (Part 1: Open coding)"
---

In this project, you will build a simple chatbot and evaluate it using the AI Evals process.

Your chatbot should specialize in some topic. For example, it could be:

* A math tutor for elementary students
* A mentor for caring for tortoises
* Your study buddy for this course!
* A chatbot that does one of the above while talking like a pirate

Do *not* attempt to make a general chatbot like ChatGPT or a general coding chatbot.  You need to narrow the scope for what the chatbot is expected to do if it is going to be successful.

Use the Llama LLM hosted on Rosie:

```
import openai
stream = False

client = OpenAI(
   base_url = 'http://dh-dgxh100-2.hpc.msoe.edu:8000/v1',
   api_key = "not_used" # this field needs to be included but is ignored
)
chat_completion = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{
        "role": "user", 
        "content": "Why is MSOE the best school to study CS and AI?"
    }],
    stream=stream
)

print(chat_completion.choices[0].message.content,sep='',end='')
```

And use this basic chatbot structure:
```
import gradio as gr
import json
messages = [{"role": "system", "content": "You are a friendly assistant"}]

try:
    with open('conversations.json') as file:
        conversations = json.load(file)
except:
    print('No conversations file yet. Starting with empty list of conversations')
    conversations = []
conversations.append(messages)

def echo(message, history):
    global messages
    global conversations
    try:
        messages.append({"role": "user", "content": message})

	# TODO: Call LLM to get an actual response
        response = "I don't do anything yet."

        messages.append({"role":"assistant","content": response})
        conversations[-1] = messages

        with open('conversations.json','w') as file:
            json.dump(conversations,file)

        return response
    except Exception as e:
        return e.message

# echo("Help me with A",[])
demo = gr.ChatInterface(fn=echo, type="messages", examples=["Help me with A", "Do B for me", "Help me with C"], title="Simple Chatbot")
demo.launch()
```

After you run the cell with this code from Code Server within a Jupyter Notebook, click on the `http://127.0.0.1:1234` link, where 1234 will be replaced by some other port number. (You will see "Unable to connect" below this, but you can ignore that.  The link above it should take you to a working version of your chatbot.)

If you see a red "Error" bubble in the middle of the chat, return to the notebook and look below the large "Unable to connect" message. There you should see the error traces. Though you may also wish to debug your function by calling it directly and commenting out the demo for a moment:

```
echo("Help me with A",[])
# demo = gr.ChatInterface
# demo.launch
```

Make the chatbot use LLM calls by splicing the provided LLM code into the provided chatbot code. Note that the provided chatbot code already has some example code to help you manage the conversation with the chatbot.

Record ten conversations with the chatbot.  Then perform open coding on each conversation.  Put the json blob for each conversation as its own line in a CSV file and use Excel to fill in a second column with your open codes.  How could your chatbot improve?  When you are done, you should have ten conversations (perhaps with slightly different versions of your chatbot's system prompt) and ten open codes, one for each conversation.  The open codes should be a short text describing very briefly what the chatbot could have done better for that conversation.

Try adjusting your system prompt after each conversation to make your chatbot better even as you generate responses. The better your chatbot can perform in this initial stage, the better the baseline that you can iterate on in future renditions of the lab.

Also, if your chatbot doesn't seem to be performing well, consider narrowing the scope of your chatbot.  Try to collect ten traces within a narrow scope where the chatbot at least performs OK, and hopefully where you think it can be improved.

Bear in mind that we *cannot* expect even the leading chatbots from the top providers like Anthropic or OpenAI to be a drop-in replacement for anything an ordinary person could accomplish.  But chatbots *can* often perform tasks reasonably well that no deterministically-programmed system could in the past.  Try to lean into the chatbot's ability to process words, carry on a conversation, etc. based on their meanings and steer away from open-ended tasks where the chatbot needs to perform near human level at many unrelated tasks to be able to perform well.
