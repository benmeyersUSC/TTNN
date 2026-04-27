# TTTN Journal

### April 26th, 2026 -- Byte Pair Tokenization

Today I implemented **Byte Pair Tokenization** for language modeling in***TTTN
***. It is both something that will be very useful for future projects created with this library and that is beautiful in its own right. It is langauge modeling to its core, and it runs lightning fast and obeys a wildly simple principle.

When I first learned of the algorithm, the thing that sucked me in --- and still holds me, fires me up --- is the notion of the algorithm's rediscovery of the existing linguistic structures we see in text. There are common morphemes, common words; in the full expression of Unicode there are emojis and other multibyte characters that, if useful, will find their way to token-hood. Decades of digital computation brought all of our symbols into a single encoded format. We learned to break down structure into bytes in an interpretable and mandated way, then we watch as the rules of dummy statistics rediscover the latent structure.

It is a constrained task in language modeling, but it contains nuggets of the most beautiful principles expressing themselves in trillion parameter language models taking over corporate America; it also vitally prepares the
*input*
for these gargantuan Mandaraxes. BPE shows how much preliminary juice is to be squeezed from statistics in language. It is an act of entropy reduction; it is Information Theory 101; it seeks to minimize the symbol count. It simply finds structure. 