---
title: "Project: Training an LLM (Part 1: Training an LLM on synthetic data)"
---

This is the first week in a multi-week lab on training LLMs.

[nanoGPT](https://github.com/karpathy/nanoGPT) is an implementation of a transformer training architecture that tries to be fairly simple while also being high-enough performance to train a GPT-2 model out of the box.  Trained on the works of Shakespeare (1MB of training data), it can generate some plausible old-English text.

In this lab sequence, you will train a transformer to produce chain-of-thought evaluation of arithmetic expressions in a very specific format.

# Rosie Setup

You will likely need Rosie for this assignment, although you could also try a colab notebook again.

On Rosie, you may want to use the conda environment set up for CSC 4611:

```
conda activate /data/csc4611/conda-csc4611/
```

The [instructions for setting up VS Code on Rosie](http://example.com) may also be useful here.

# Chain of thought

When you ask ChatGPT to answer a question for you right away, it has to know to produce the right answer on the first token.  But if you ask it to produce several steps and then give you the answer, it can sometimes calculate a better answer.  This is particularly true for many-digit arithmetic:

For example, when [asked to evalute](https://chatgpt.com/share/68dede3b-2eb0-8001-bf9e-2edd9c8a1560) (((4 + 5) * (3 - 2)) + (1 * 7) + ((3 + 8) * (2 - 1))) immediately without breaking the task down into steps, even with a recent version of ChatGPT (Oct 2025) does not generate the right answer. (It answers 24, when the correct answer is 27.) But when given a chance to work through the problem step by step, it comes to the right answer.

Why is this?  As the transformer generates tokens one at a time, it builds a context window so that by the time it is generating the last token, it has all of the intermediate steps as context building towards that correct answer.

This is chain-of-thought reasoning.  Whenever a model works through a question step-by=step, it is generating a "chain of thought" leading from the question to an answer.  For mathematical questions, chain-of-thought reasoning can greatly improve the reliability of the model. (For more subjective questions, chain-of-thought reasoning can actually further reinforce the unhealthy biases the model picks up from society.)

Even the so called "reasoning" models are simply trained to use this strategy without being prompted to do so.  In the chat above, if you click on the "Thought for 17s" link, you will see that there is a reasoning chain shown on the right.  This reasoning chain is simply more tokens the model generates to give itself more context when starting to generate the output tokens.  Thease are sometimes called "reasoning" tokens because the model may generate steps in a rational chain leading towards a correct answer. Others call these "intermediate" tokens. But whatever you call them, they are another example of chain-of-thought reasoning.

There are many ways to get a model to reason in chain of thought.  Perhaps the simplest way is to instruct the model to think step-by step in the prompt. Since there are many examples on the internet where people reason step-by-step, models pretrained to extend internet text already know how to reason step-by-step.

Or, you could fine-tune a model. After instructing it to think step by step on many prompts, you could remove the "think step by step" instruction and just give it the question and its own step by step reasoning from the earlier prompts as further training data.  After seeing many examples where the "correct" extension of the text was to reason step by step, the model will naturally start producing reasoning steps rather than the correct answer, since this is what it was re-trained to do.

# Chain-of-thought pretraining

But in this lab sequence, we are going to take a different approach.  We are going to train a model to reason in chain of thought step-by-step from the start, without relying on internet-scale pretraining.  So the ONLY thing our model will be able to do is chain-of-thought reasoning in the format we teach it to do, just like when training a model on Shakespeare, the only thing it can do is extend a Shakespeare text.

Since the first training of a model on a corpus is called "pre-training", we will call this chain-of-thought pretraining.

# Our chain of thought environment 

Consider a purely-nested arithmetic expression, where each set of parenthesis contains only one operation on two other parenthetical expressions.  For example, the expression we gave to ChatGPT earlier on could be expressed as:

`((((4 + 5) * (3 - 2)) + (1 * 7)) + ((3 + 8) * (2 - 1)))`

One way to evaluate such an expression is to only evaluate one innermost set of parentesis at a time.  When repeated, this leads naturally to a chain-of-thought text, leading to an answer:

```
((((4 + 5) * (3 - 2)) + (1 * 7)) + ((3 + 8) * (2 - 1)))
=(((9 * (3 - 2)) + (1 * 7)) + ((3 + 8) * (2 - 1))) 
=(((9 * 1) + (1 * 7)) + ((3 + 8) * (2 - 1))) 
=((9 + (1 * 7)) + ((3 + 8) * (2 - 1))) 
=((9 + 7) + ((3 + 8) * (2 - 1))) 
=(16 + ((3 + 8) * (2 - 1))) 
=(16 + (11 * (2 - 1))) 
=(16 + (11 * 1)) 
=(16 + 11) 
=27 
```

If we train an LLM on many examples like this, it should learn to extend the text by generating, one token at a time, the sort of text we see here.

In this week's lab, you are will generate many many examples like this.

For example, I generated text containing examples like these:

```
Expression Evaluation Process
==================================================

Step 1:
Expression: (13 - 76)
Evaluating: (13 - 76) = -63
------------------------------

Step 2:
Expression: -63
Final Result: -63
------------------------------

Expression Evaluation Process
==================================================

Step 1:
Expression: (((((41 * 71) + 86) + (84 * 13)) + 87) - ((((90 - 87) + (81 + 97)) + 76) + (((22 * 97) * (59 - 51)) * ((10 * 77) + 1))))
Evaluating: (41 * 71) = 2911
------------------------------

Step 2:
Expression: ((((2911 + 86) + (84 * 13)) + 87) - ((((90 - 87) + (81 + 97)) + 76) + (((22 * 97) * (59 - 51)) * ((10 * 77) + 1))))
Evaluating: (90 - 87) = 3
------------------------------

[...]

Step 17:
Expression: (4176 - 13162769)
Evaluating: (4176 - 13162769) = -13158593
------------------------------

Step 18:
Expression: -13158593
Final Result: -13158593
------------------------------

Expression Evaluation Process
==================================================

Step 1:
Expression: ((((84 - (15 + 56)) - ((5 + 22) * (51 - 44))) + 87) * (((63 - (1 - 17)) + 79) - (((44 + 46) - (12 + 72)) * ((53 * 75) - (75 + 64)))))
Evaluating: (15 + 56) = 71
------------------------------
...
```

# Vibe coding

Vive coding is the art of guiding an LLM to produce the code you want.  (Although some say to truly "vibe" code, you need to learn to go with the flow, get in the "vibe", and hardly look at the source code at all!) 

Currently, not all coding tasks are best vibe-coded, but this week's lab is a perfect example of task that makes sense to vibe code. (While the most care-free sense of vibe-coding might work for this lab, as we will discuss later, you WILL want to look closely at the output produced by your code, even if you don't look at the code itself!)

If you have taken a course like Data Structures, you probably are familiar with the concept of an expression tree, and, given a good deal of time, you could probably write not only code to evaluate expression trees, but to generate random expression trees as well. So could I. (Well, at least, I hope we could.)

But this is a task that is *perfect* for an LLM.  It can generate this code easily, often with only a very brief text prompt.

That's why your homework task this week is to guide an LLM agent into generating the code needed to generate a dataset of chain-of-thought parenthetic-expression evaluation examples.

# Selecting an agent

You can use whatever agent you want for this lab.

I used the free version of cursor.  But you could also use VS Code with the Github Copilot plugin (I think you can also get this free as a student, perhaps).  Or you might try the trial for JetBrain's Junie.

# Assignment

Select an agent. Instruct the agent to generate code that will generate examples similar to the chains of thought shown earlier in the lab.  You do NOT need to get the agent to follow either example format exactly. 

The format shown above works okay as a chain-of-thought format, but you may find that a different format makes it easier for an LLM to learn the task.  And, for future labs, you may want your LLM to work in a dense format which will generate fewer tokens but still come to a correct answer.  (This can make examining the chain-of thought reasoning easier)

Instruct your agent to generate a thousand chain of thought examples.  This should generate a file of several thousand lines.

Take an LLM training setup, such as the nanoGPT setup mentioned in the introduction to this lab sequence. (A follow-on lab builds fairly closely on the nanoGPT setup, so you may wish to use it to make that code work better. But you are free to also use any other LLM training codebase that you wish.)

Get it to train on a provided text of at least a million tokens (such as the works of Shakespeare or the KJV.)  Train character-by-character.  After training, confirm the LLM can generate text with real English words or mostly-sensible English sentences, as in the examples given.

Then replace the text with your chain-of-though text from earlier in this lab.  Train until the model can generate the general format of your chain-of-thought reasoning.

Then try to train further to see how complex of instructions it can evaluate.  Modify the script to allow resuming an existing training session. This can allow you to train your model a while, see how well it is working, and then start training again.

To test your model's ability to do inference, insert simple parenthetic expressions in the format your model produces, and generate enough tokens for the model to work through your chain-of-thought reasoning and produce a response (you might look for keywords in the output that would start the next example as the condition to terminate generating new tokens).

For example, if your transformer generates output like

```
Expression Evaluation Process
==================================================

Step 1:
Expression: (((((91 + 65) * (75 * 79)) + ((60 - 96) + 27)) + 81) * 52)
Evaluating: (91 + 65) = 156
------------------------------

Step 2:
Expression: ((((156 * (75 * 79)) + ((60 - 96) + 27)) + 81) * 52)
Evaluating: (75 * 79) = 5925
------------------------------

[...]

Step 8:
Expression: (924372 * 52)
Evaluating: (924372 * 52) = 48067344
------------------------------

Step 9:
Expression: 48067344
Final Result: 48067344
------------------------------

Expression Evaluation Process
==================================================

Step 1:
Expression: ((62 - 42) - ((((72 * 56) * (12 - 77)) + ((95 * 23) + 87)) - ((18 + (79 - 47)) - ((70 + 20) + (24 * 22)))))
```

You can try evaluating it by starting to generate with text like

```
Expression: ((2 + 1) * 5)
```

If things are going great, it should extend this text toward a calcuation result like:

```
Expression: ((2 + 1) * 5)
Evaluating: (2 + 1) = 3
------------------------------

Step 2:
Expression: (3 * 5)
Evaluating: (3 * 5) = 15
------------------------------

Step 9:
Expression: 15
Final Result: 15
------------------------------

Expression Evaluation Process
```

And you might choose to stop generating text when the text "Evaluation Process" occurs within your generated text.

As you train, consider adjusting your vibe-coded arithmetic expression code to aid the model in learning.

# Report

Submit a VERY brief report called `report.txt` in markdown format and the .py file created by your agent.

In your report, give an example of one of the prompts you used while vibe-coding the agent.  (It might even be your ONLY prompt!)  Also give an example of a chain of thought text produced by the code and report on the total number of chains of thought you generated.

Submit:
* The .py file created by your agent to produce the expression trees
* Any .py files you edit from whatever repository you work with. (Git can be helpful for determining what files these are)
* A brief `report.txt` in markdown format which includes:
  - A description of what model size you used
  - An example of one of the prompts you used while vibe-coding the agent
  - An example of a chain of thought text you used to train your model.
  - A description of your training process, summarizing any paramters you adjusted while trying to get the model to train for you (e.g. number of epochs, learning rate, etc.)
  - A brief discussion of whether you changed the chain of thought format produed by the vibe-coding and why or why not
  - A description of some metric recorded during training of your final model such as loss or next-character-prediction accuracy
  - Some examples of simple expressions and the chain of thought your model produced while evaluating them.

(Because the examples may be lengthy, your report may be a little bit less brief)

Try to get your model to evaluate as complex of expressions as it can.  It almost surely won't be able to evaluate everything it saw in the training set.   If it can evaluate simple arithmetic expressions that is great.  Please leave time so you can iterate a bit trying to get a model that trains and evaluates at least one VERY simple expression.

Again, consider adjusting the chain-of-thought format produced by your vibe coding to see if it helps you to train a model that reasons through chain of thought prompting more reliably.

# Acknowledgement

The idea of training an LLM on chain of thought nested arithmetic expressions reasoning was proposed by Jonny Keane, one of the first MSOE MSML grads.
