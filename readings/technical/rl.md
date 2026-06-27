title: "Reading: RLHF and DPO (Reinforcement Learning for LLMs)"

# Reinforcement Learning

Reinforcement learning algorithms are trained to maximize the reward they receive after making many decisions (or actions).

When applied to transformers, a reinforcement learning algorithm received a single reward after generating many tokens (that is, after it has completed answering a question).  This sets reinforcement learning algorithms distinctly apart from supervised fine tuning.  Rather than having a "right" signal at every toke generated (because the tokens are generated in batch, and only predict one token at a time), the model must generate many tokens autoregressively, exploring the space of possible text by generating many tokens determined by its own output as well as the prompt.

How can a model learn to change a token early in the sequence that lead to an incorrect output later in the sequence?  That's the sort of question that we will explore in this week's reading.

# Reinforcement Learning

There is SOOO much to learn about reinforcement learning. 

There is an interesting interplay between Reinforcement Learning and Deep Learning.  As mentioned above, Deep Learning can be a good tool to make classic Reinforcement Learning algorithms start to work -- model-based learning in particular.

But at the same time, Reinforcement Learning has become a very important tool to build the best deep learning models as well. And that is our focus this week.

One of the most popular reinforcement learning algorithms today is PPO (Proximal Policy Optimization).  This algorithm is popular because it is able to learn robustly on a wide set of problems without needing to tune the reward signal and hyperparameters used as much as is required with other reinforcement learning algorithms.  PPO is an actor-critic approach. It uses two models: the actor (or policy) model, and the critic (or value) model.

When applied to RLHF, the PPO algorithm makes use of three models:

* A reward model, which provides a reward for a particular input and complete answer
* A value model, which predicts the reward that a partial answer will get once it is complete, token by token
* A policy model, which is just a transformer that generates an "action" of generating the next token 

But some variations on PPO for RLHF "simplify" this to a "bandit" or one-step process.  In this case, the three model are:

* A reward model, which provides a reward for a particular input and complete answer (just like in the step-by-step version above)
* A value model, which predicts, given just the prompt, the average reward that a complete response will get for this answer by sampling through to completion with the current model.
* A policy model, which is just a transformer that generates an "action" of generating a complete response to the model.

# A note on reading probability math:

When you see an expression like 

$$(y_{1}, y_{2}) \sim \pi^{SFT}(y | x)$$

you should read this as "samples $y_{1}$ and $y_{2}$ are drawn from the random distribution called $\pi^{SFT}.$ You can think of $\pi^{SFT}$ as a function that takes in $x$ (and everything after the bar) and outputs $y$ (and any other variables before the bar), adding some randomness. In this case, the rand function is called twice, and $y_{1} and y_{2}$ are two results of different random values gotten by calling the random function with the same $x$.  In the context of the paper, $x$ is the prompt and the two $y$'s are two different outputs of the LLM for the same prompt.  (Though sometimes probabilistic algorithms use the mean value of a distribution so they don't add any randomness to their outputs.)

The probability of a response $\pi_\theta(y|x)$ is simply the product of the softmax value for each token in the generated text. (You can express this mathematically as $\pi_\theta(y|x) = \pi_\theta(y_{T}|y_{T-1},y_{T-2},..., y_{1}, x) \cdot \pi_\theta(y_{T-1}|y_{T-2},..., y_{1}, x) \cdot \pi_\theta(y_{T-2}|,..., y_{1}, x) \cdot \pi_\theta(y_{1}|x)$) 

As Claude helps us to understand:

> Suppose we have:
> - Prompt x = "The cat"
> - Response y = "sat down" (three tokens: "sat", "down", "<EOS>")
> 
> Then:
> ```
> π_θ(y|x) = π_θ("sat" | "The cat") 
>          × π_θ("down" | "The cat sat")
>          × π_θ("<EOS>" | "The cat sat down")
> ```
> 
> If the model's softmax probabilities are:
> - P("sat" | "The cat") = 0.4
> - P("down" | "The cat sat") = 0.6  
> - P("<EOS>" | "The cat sat down") = 0.8
> 
> Then: π_θ(y|x) = 0.4 × 0.6 × 0.8 = 0.192
> 
> In practice, we work with log probabilities to avoid numerical underflow:
> ```
> log π_θ(y|x) = log(0.4) + log(0.6) + log(0.8)
>              = -0.916 + (-0.511) + (-0.223)
>              = -1.650
> ```
> 

But note that when we write $(y_{1}, y_{2}) \sim \pi^{SFT}(y | x)$, we are *not* talking about finding the probability that the model would have generated $y_1$ or $y_2$. Instead, we are saying that we are using the model to generate $y_1$ and $y_2$.  That's what drawing from the distribution of the model means here.  So throughout this reading, $\pi_\theta$ can refer to either feeding an existing text through the model to find its probability or using the model to generate a new response.  In the latter case, you will always see $\sim$ used.

Similarly, when you see an expression like

$$\mathcal{L}_R(r_\phi,\mathcal{D}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} [\log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l)]$$

you can think, "thank you for sharing your code with me," and then read it like this:

The loss $\mathcal{L}$ that we use to train the reward model $r_\phi$ on the dataset $\mathcal{D}$ is computed like this:

We take a sample $(x, y_w, y_l)$ from the training dataset $\mathcal{D}$ where $x$ is the prompt and $y_w$ and $y_l$ are the winning and losing LLM outputs for this prompt, and feed the outputs through the reward model to find $r_\phi(x,y_w)$ and $r_\phi(x,y_l)$.  We then take the difference and feed them through a sigmoid function $\sigma$ and take the log to find the "log-likelihood loss".  We then use this loss as something to minimize in backpropagation to train the network that calculates $r_\phi(x,y)$, updating the weights in $r(...)$ to make the loss as small as possible. (The \mathbb{E} here stands for "expectation" and it simply means that we will find the average loss over every sample in a batch, or, more broadly, over every sample in the training set, though when training a network, we may very well be looking at the loss at only one sample at a time if our batch size is 1.)

# Direct Policy Optimization

Jump to section 3 (Preliminaries) of the [Direct Preference Optimization (DPO) paper](https://arxiv.org/pdf/2305.18290).  Note that this section is NOT yet talking about DPO, but rather summarizing how RLHF is done.

**1. What do the inputs x, y1, and y2 to the preference model `p*` mean?**

**2. What are the inputs to the reward model `r*(x,y)` or $r_\phi(x,y)$? Why are these different than the inputs to the preference model?**

Consider the loss fuction for training the model in equation 2.

**3. What values are in each sample drawn from the dataset $\mathcal{D}$?**

**4. What model is being trained in equation 2?**

**5. How is the loss calculated once we know the output of the model for the two samples?**

Consider equation 3, the training objective used to train the language model (that is, the policy model) during fine-tuning.

**6. From what dataset is the prompt $x$ selected when evaluating the optimization in eq. 3?**

**7. From where does the response $y$ come when evaluating the optimization in eq. 3?**

**8. Do we contrast the reward for two responses to a prompt in eq 3. or just use a single reward? Why is this important for reinforcement learning?**

Note that the goal of PPO is to optimize Eq 3, but we can't do this directly. Instead, during PPO, we will alternate between performing backpropagation on the policy model (that is, the language model we want to train) and backpropagation on the value model (which is also a language model, but is only used to estimate how good of a reward we ought to get before the response is completed.) Nevertheless, the answers you wrote to questions 6-8 above still apply to the process of training a model with PPO.

Now read through section 4 to equation 7.  Focus on the text, not the equations, until you reach equation 7.

Consider equation 7, which is the loss function used to train DPO.  Unlike equation 3, we can perform backpropagation on equation 7 and thus train a model using this as the loss function.

**9. What values are in each sample used to train DPO?** 

**10. When training DPO, do we contrast two responses or just use a single response for each prompt?**

**11. Summarize the differences between training RLHF with PPO vs training using DPO.**

As Claude says...

> $\log \pi_\theta$ is exactly what the model already computes during forward passes! During supervised fine-tuning, you compare these log probabilities against the "correct" tokens. During DPO, you compare the log probabilities of the winning response against those of the losing response and the reference models.

The DPO loss is very easy to do backpropagation on!  We simply feed the entire generated prompt into our model, get the softmax predictions for every token, and then do exactly the same backpropagation process that we would do during supervised fine tuning.  Except that, in this case, we are not comparing the output with "known" tokens. Instead, we care about how the backprop for the winning response compares with the backprop for the losing response -- and how both of those compare with the reference models.

Some students wonder how DPO is actually implemented.  I want to emphasize: Equation 7 (on p. 4) in the paper IS the implementation of DPO: Feed the text of the winning and losing responses into the model. Take the log of the softmax probabilities for each of the correct tokens (with the output shifted one token to the right as usual.). Sum them. This gives you ($\log \pi_\theta(y_w|x)$ and $\log \pi_\theta(y_w|x)$. Do the same with the reference model.  Multiply by $\beta$, stuff it through the sigmoid function ($\sigma(..)$) and another log function.  This is your loss. Use an autograd library to find the gradients of all your model parameters -- that is, the way you change these parameters to minimize the loss. Give the gradients to your favorite optimizer (SGD, AdamW, etc) for training. Train over all the paired samples in your training set. Do all this and you've done DPO.

**12. What are the key details of the implementation of DPO?**

Look at the equation in the paragraph on p. 5 titled "**What does the DPO update do?**".  This is the gradient that an autograd library will automatically find when you give it equation 7 as discussed in the previous question.  Notice that this equation includes EXACTLY the same term we would have in supervised fine tuning: $\nabla_\theta \log \pi(y_w|x)$ is just the classic cross-entropy loss that compares the probability that the network assigns a token with the correct answer determined by shifting the example tokens one to the left. Similarly  $-\nabla_\theta \log \pi(y_l|x)$ is just the opposite of SFT: It simply trains the network to NOT produce the wrong answers.   Finally, the term $\sigma(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w))$ is a kind of boosting term: It's a term that turns off the gradient if the model is already more likely to produce the winning response.

Knowing this can build your intuition about what RL can and cannot do -- its strengths and weaknesses are similar to pretraining and supervised fine-tuning. Given enough example texts, RL can infer deeper structures that generalize principles about what to do and not do in a response.  But a small amount of RL training will simply adjust the model's biases to make the winning responses more likely.

**13. Describe the similarities and differences between supervised fine tuning and DPO.**

# Optional extras
If you are wanting an organized guide to all the terms used in the reinforcement learning literature,  Dr. Kedziora recommends [the RL Book](http://www.incompleteideas.net/book/RLbook2020trimmed.pdf) by Sutton & Barto, and I (Dr. Yoder) have also found it helpful -- it is even cited by [leading research (Dreamer V3)](https://arxiv.org/pdf/2301.04104) on using deep learning (primarily CNNs) to support reinforcement learning goals.

Dr. Kedziora also teaches an entire elective on reinforcement learning. We are just at the point in the term where they are going over policy-based algorithms (which all RLHF use), actor-critic algorithms (that is, policy-value algorithms, which PPO is an example of) and PPO.  I highly recommend this course to get a deeper theoretical understanding of these algorithms.

But for the purposes of this reading, we need to have a very simple understanding of RL and focus on how it plays with LLMs.  **So I don't expect you to even skim the two links in the paragraph above.**  I mention them only to emphasize that the RL Book can be a good resource for looking up what a term means if you are confused about it.

## More optional extras

There are two GRPO algorithms in deep learning, both well-established:

* [Group Robust Preference Optimization in Reward-free RLHF](https://arxiv.org/pdf/2405.20304)
* [Group Relative Policy Optimization (GRPO), as introduced in DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)

The second (DeepSeek) one is the one I wish I had time to share with you this term.  Like DPO, DeepSeek eliminates the need to train a separate value function. But it still uses the clipped updates of the PPO algorithm and is an algorithm that is useful for RLVR training. The idea of not using the value function is one that companies have also found effective as well.

Finally, the ideas of RL will likely continue to influence the way we think about fine tuning LLMs. A recent work in this space is inspired by the idea of building world models. It appears to be a technique somewhere between supervised fine tuning and RL in which models learn not from a fixed corpora but from generating commands and seeing how their environment responds to them: 
[\[2510.08558\] Agent Learning via Early Experience](https://arxiv.org/pdf/2510.08558) - something between SFT and RL. Doesn't use rewards, but IS softly steered by human examples.
