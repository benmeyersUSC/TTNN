***ProCC Compiler***

Use TTTN to train an ENC-DEC Transformer on the task of compiling code -> assembly from TAC-435 ProCC compiler.

Agenda:

- ProCC side
    - Need fixed vocab, capped length, capped complexity program generator
    - Compile all valid programs, only keep assembly with capped length
    - For both, we'll want to tokenize to a fixed (shared?) vocab and <PAD>
        - ints have <INT0>, <INT1>, ..., <INT9>
        - variables will be like <VAR><INT>
        - keywords will be like <IF>
        - assembly ops will have their own tokens
            - <ADD>, <PUSH>, etc
            - for registers <REG><INT>
- TTTN side
    - Generalized ENC-DEC Transformer, given token vocab size, input/output windows
    - ENC
        - embed, positional encode, bidirectional attention
        - return residual stream + final attention for context
    - DEC
        - take in ENC stream + context
        - causal mask attention
        - training:
            - use all outputs
        - inference:
            - autoregressively generate