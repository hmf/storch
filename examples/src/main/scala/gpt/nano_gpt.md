
# [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Links

1. [Google Colab code](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
1. [nanoGPT lecture](https://github.com/karpathy/ng-video-lecture)
1. [nanoGPT repo](https://github.com/karpathy/nanoGPT)
1. [minGPT repo (deactivated)](https://github.com/karpathy/minGPT)
1. [my website](https://karpathy.ai)
1. [my twitter](https://twitter.com/karpathy)
1. [our Discord channel](https://discord.gg/3zy8kqD9Cp)
1. [Playlist of the whole Zero to Hero series](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

## Supplementary links:
* Attention is All You Need paper: https://arxiv.org/abs/1706.03762
* OpenAI GPT-3 paper: https://arxiv.org/abs/2005.14165 
* OpenAI ChatGPT blog post: https://openai.com/blog/chatgpt/
* The GPU I'm training the model on is from Lambda GPU Cloud, I think the best and easiest way to spin up an on-demand GPU instance in the cloud that you can ssh to: https://lambdalabs.com . If you prefer to work in notebooks, I think the easiest path today is Google Colab.

## Background

1. [The spelled-out intro to language modeling: building `makemore``](https://www.youtube.com/watch?v=PaCmpygFfXo)

## Tokenizers

1. [Unsupervised text tokenizer for Neural Network-based text generation](https://github.com/google/sentencepiece)
   1. [natural-language-processing](https://github.com/topics/natural-language-processing)
   1. [neural-machine-translation](https://github.com/topics/neural-machine-translation)
   1. [word-segmentation](https://github.com/topics/word-segmentation)
1. [TikToken fast BPE tokeniser use by OpenAI](https://github.com/openai/tiktoken)

## Notes

1. [01:16:56](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4616s) note 6: "scaled" self-attention. why divide by sqrt(head_size)
1. To improve the network:
   1. [01:26:48](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5208s) residual connections
   1. [01:32:51](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5571s) `layernorm` (and its relationship to our previous `batchnorm`)
   1. [01:37:49](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5869s) scaling up the model! creating a few variables. Adding `dropout`
   