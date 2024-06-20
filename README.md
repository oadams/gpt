# A basic generative pre-trained transformer

This repository contains a basic implementation of a generative pre-trained
transformer (GPT). The implementation was inspired by and closely follows the
implementation in [Let's build GPT: from scratch, in code, spelled
out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy. This
implementation extends on the material in that video by replacing all
references to `torch.nn` with our own implementation. This includes
implementations of things such as:
- activations like softmax and GeLU
- `Module`s and `Parameter`s
- Weight initialization
- Linear and Embedding modules
- Cross entropy loss function
- Layer normalization
- AdamW optimizer
- Dropout
The implementations are intended to be for educational purposes. Though the
interface for our purposes matches that of the `torch.nn` implementation, such
that the PyTorch implementations can be seamlessly substituted in place of any
of our modules, the PyTorch offering may be more general and other features and
will often be faster.

We still use PyTorch, most crucially for the implementation of tensors and
their operations, as well as autograd.

Sitting down and implementing such things forces oneself to work through
inevitable gaps in knowledge.

The intention is for this to be a repo for me to come back to and add to when I want to
more intimately understand a technique I haven't already implemented. Once I
implement a given module it gets put in my own personal spaced repetition
system so that I do not forget it. Transformers have been around long enough
that it's clear it's just worth understanding exactly how each of the sub
compoments works, and not forgetting the details.

There are plenty of things to do, just off the top of my head:
- [ ] Continue to replace `torch` functions with our own, including our own
  autodifferentiation implementation and writing CUDA kernels for tensor
  operations.
- [ ] Continue expanding the implementation to things such as ROPE embeddings,
  Grouped Query Attention, Key-Value caching, quantization, data loaders,
  distributed training, and so on
  essentially ad infinitum.
- [ ] Add perpexity calculations

And as far as code quality goes, just some starting notes:
- [ ] More clearly document the code
- [ ] Structure the code as a python package properly, do some linting
- [ ] Extend the config module so that `config.toml` can be used to easily sub
  in `torch.nn` implementations. This exists for `Module` and `Parameter`, so
  it's just a matter of extending it to other implementations.
