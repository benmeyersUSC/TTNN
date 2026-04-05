***Mapping Latent Spaced***

Use TTTN to define two autoencoders of exactly the same type, trained on exactly the same dataset. Observe latent space and differences in absolute values of the networks. Freeze Enc/Dec and add a single trainable linear map. Run inference from A -> A, A -> B, B -> B, B -> A, A -> Map_ab -> B, B -> Map_ba -> A. 