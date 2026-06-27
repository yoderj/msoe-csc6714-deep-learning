---
title: "Project: AI Evals on a Simple Chatbot (Part 2: Axial coding and evals)"
---

In this project, you will build a simple chatbot and evaluate it using the AI Evals process.  

Start a report.txt in markdown format to report your findings from this part of the lab.

In the previous part, you built and open-coded ten responses from a chatbot. In this part, you will axial code those ten responses, identify one issue in which you would most like the bot to improve, then develop and test an AI eval for that issue.

1. Copy and paste your open codes into a chatbot (e.g. ChatGPT) and ask it to categorize them into three to five axial codes. Include the list of codes it gives you in your report.
2. Take the list of codes (error categories) that ChatGPT returns and refines them. Do any of the codes duplicate other codes? Get rid of them. Is a code to broad? Break it into finer codes.  Include your refined list of codes and briefly explain the most important edit you made to the list.
3. Take your refined list of codes, and ask ChatGPT (or the Rosie llama model) to assign an axial code for each of your open codes. Copy these into a new column of the spreadsheet you started last week.  Summarize counts of each axial code in a table in your report.
4. Considering your summarization, where would you like the codes most to improve?  Which code is most important?  It doesn't have to be your most frequent code. Is there an aspect of this code that you would like to start with?  Define a specific LLM-as-judge or programmatic evaluation of your system. This could be detecting whenever that most important code ought to be applied. But it could also be a special case of that code. Briefly describe this process in your report.  Do *not* simply wrap the axial code's name or even a brief description of the axial code into a generic LLM prompt.  Instead, think about what sort of guided chain-of-thought prompting might be effective to help an LLM understand what you are looking for with that prompt.  If you don't have ground-truth for the evaluation (and we usually don't), be especially clear about what the issue is that you want the evaluation to identify. 
5. Create a boolean column for your previous conversations -- e.g., true when the error occurs and false when it does not. Summarize how many trues and falses are in this column in your report.
6. Write an LLM program that, iterating over your CSV, also generates a true-when-error-false-when-not column, using your evaluation as the baseline.
7. Compare these two columns (e.g., with accuracy, true positive rate, and false positive rate, or accuracy, precision, and recall) and provide these summaries in your report. 
8. Tweak your evaluation to improve its performance. Make a note in your report on how you are changing the prompt and why you hope it will improve the result.
9. Run your revised evaluation, generating another true-false column. Does it perform better or worse? Comment on this in your report.
10. Edit your system prompt again to try to improve your chatbot in the area that your single eval evaluates.
10. Run some more traces with the new system prompt run your best eval on them. Has your chatbot improved in this area?

Submit your report.txt in markdown format and your .ipynb or .py file for your chatbot.
