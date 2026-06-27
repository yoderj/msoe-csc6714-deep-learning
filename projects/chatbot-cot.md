---
title: "Project: AI Evals on a Simple Chatbot (Part 3: Reasoning chatbot)"
---

In this project, you will build a simple chatbot and evaluate it using the AI Evals process.  

Start a report.txt in markdown format to report your findings from this part of the lab.

So far, your chatbot has had a system prompt, but all of its output tokens have gone directly to the user.  Here, we are going to allow your chatbot to reason.

For LLMs, "reasoning" is simply generating tokens that the user doesn't see (or, if they can see them, that are collapsed in a "reasoning" section of the output).

To make your chatbot reason, simply adjust the system prompt to instruct the chatbot to think through what it's going to say and then end with

```
<response>
</response>
```

tags, with the output to the user within the the tags. You will need to modify your chatbot to take the output of the LLM and extract this response, only passing along what is within the `<response></response>` token to the user.  If your parser fails to find the response tags, call the LLM again (up to three times) and return the first successful response or a message that an error has occured.

Now, test your LLM again -- does it perform better or worse at your key evaluation axis compared to last week?  If both last week's and this week's evaluation are working perfectly, consider picking up another axial direction and developing an AI eval that works in that direction.

Try a variety of CoT system prompts, guiding the LLM to reason in a variety of ways in an attempt to get it to reason better before responding ot the user. Can you get your chatbot to a workable sense this way?

Again, submit your report.txt in markdown format and your chatbot's source code.
