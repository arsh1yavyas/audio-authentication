# Initial Research on Voice Cloning

- From Less Data + Faster + Lower Quality to More Data + Slower + Higher Quality:
  - Zero-shot (no training, just inference and reference audio provided at generation time)
  - Few-shot (minimal adaptation of general speech model and minutes of target speaker audio)
  - Fine-tuning (adapt pre-trained, general speech model to specific voice and hours of target speaker audio)
  - Training from scratch (no pre-trained knowledge and weeks/months of audio)
  - Training from scratch is rarely used in modern voice cloning since pre-trained models have learned general speech and transfer learning is much more efficient
- VC models learn to separate linguistic content from speaker identity
  - They can then use the same model for outputs (same words) in several different speakers' voices by just using different speaker embeddings each time
  - Speaker embedding: vector representation of a speaker's voice that represents that voice identity with numbers ("voice fingerprint")
  - Speaker embeddings can be extracted from audio via self-supervised speech models like WavLM
  - Zero-shot and few-shot learning (both instant) train model on many different speakers and also learn to extract speaker embeddings
  - Then, once given a target text and voice, they can extract the speaker embedding and generate speech using it
  - For few-shot, having more data leads to a better embedding, which leads to higher quality cloning
- With fine-tuning, you take a pre-trained Text-to-Speech model and use hours of desired speaker audio to find specialized model weights
  - Modern approach is to use adapter layers (LoRA or Low-Rank Adaptation of LLMs) on top of the model
  - Instead of changing model weights, can change widths of the adapter matrix
