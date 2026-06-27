This is the first of a series of labs where we look at using transformers for a grocery-shopping application.

Consider `grocery_store.csv`.

Download `grocery_store.csv`.  Take the first half of the rows in this file. (There are too many names in this file to fit in the context of the LLM we will use for this part!)

Create a iynb notebook called `simple_prompt.ipynb`.

Use a single prompt to the LLM to search through the grocery store and find an ingredient for each item on this short shopping list:

* Bread
* Milk
* Broccoli

Use the Llama LLM hosted on Rosie:

```
import openai
stream = True # or False

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

if stream:
    for event in chat_completion:
        if event.choices[0].finish_reason:
            pass
        else:
            print(event.choices[0].delta.content,sep='',end='')
else:
    print(chat_completion.choices[0].message.content,sep='',end='')
```

Manually check that the items returned actually appear on the list. Report the items your script finds in a comment or text box at the end of your `simple_prompt.ipynb` file.

We will use this simple application as a baseline to which we will compare future development in this space.

-----

P.S. We have discussed how a model needs to be fine-tuned to learn the format of chat.  The reverse is also true: Once a model is fine-tuned for chat, you have to talk to it in the format it learned for it to chat properly with you. It's possible to use the Llama model in completions mode rather than chat completions mode:

```
from openai import OpenAI
client = OpenAI(
   base_url = 'http://dh-dgxh100-2.hpc.msoe.edu:8000/v1',
   api_key = "not_used" # this field needs to be included but is ignored
)
completion = client.completions.create(
   model="meta/llama-3.3-70b-instruct",
   prompt='Why is MSOE the best school to study CS?', # your prompt goes here
   max_tokens=256, #update this as needed
   stream=False #note that there are other available parameters that can be passed to this API
)
```
But DON'T use it this way. In this mode, your text will NOT be formatted as a chat session and the model may act strangely. For example, instead of answering your question, it may start asking its own questions and answering those instead of yours!


At the end of your Python notebook, answer these questions: (You can copy-paste them into a markdown cell)

```
**1. How could an LLM be improved to perform this task better?  Do we need better software surrounding the LLM?   Or is there something fundamental about an LLM that limits its performance at this task?**

**2. What does the LLM do well when performing this task?  What part(s) of an LLM enable it to do this well?**
```
