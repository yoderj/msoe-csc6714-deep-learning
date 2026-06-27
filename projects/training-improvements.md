---
title: "Project: Training an LLM (Part 2: Improving training performance.)"
---

This is the second week in a multi-week lab on training LLMs.


# Assignment

Last week, the instructions ended with this aspirational guidance:

> Try to get your model to evaluate as complex of expressions as it can.  It almost surely won't be able to evaluate everything it saw in the training set.   If it can evaluate simple arithmetic expressions that is great.  Please leave time so you can iterate a bit trying to get a model that trains and evaluates at least one VERY simple expression.

This week, you simply should continue iterating further, ideally until:

* The model can handle expressions reliably with four numbers in them.
* The model correctly performs all operations on numbers from 0 to 20

Based on both Dr. Yoder's experience and the experience of students who have already achieved this goal, here is our advice:

* Just start by trying to train your model longer!
  * Try resuming model training from one of your current models and simply training it longer to see if this will increase its performance.
  * Look at the learning rate schedule and make sure that the model doesn't have a learning rate of nearly zero for most of your training. (Whether or not you adjust it, it's good to be aware of what learning rate schedule you are using. When you continue training, you likely don't need to start from a zero learning rate like an uninitialized model does for stability reasons.)
* You should not need a bigger model, but you might find it helpful to set the dropout rate to zero.
* It might help to train on a simpler instruction set that has more examples of working with simpler numbers.

Just for fun, you may find that after the model is more reliable on simple arithmetic, you can get up to two-digit arithmetic to work fairly reliably by exploring a slightly larger model size.

Please continue training as early as possible so that if you feel like you are stuck or your model is saturating below the required performance, you can reach out to me for help early in the week.
