# NMT - Neural Machine Translation with Transformers

This project offers a framework for training, evaluating, and analyzing neural machine translation models using the Transformer architecture, with support for PyTorch's distributed training. It features dynamic parameter configuration, custom learning rate scheduling, attention weight visualization, and BLEU score computation for performance evaluation.

## Features

- **Distributed Training:** Leverage multiple GPUs for efficient training of Transformer models.
- **Dynamic Configuration:** JSON configuration files allow for flexible parameter adjustments.
- **Learning Rate Scheduler:** A custom scheduler supports optimal training with warmup and decay phases.
- **Attention Visualization:** Visualize model attention mechanisms with `plot_entropy.py`.
- **BLEU Score Computation:** Quantitatively evaluate translation quality using `get_bleu_score.py`.
- **BPE Tokenization:** Efficient text tokenization with `bpe_tokenizer_train.py` supporting Byte Pair Encoding.

## Requirements

- Python 3.8+
- PyTorch 1.7+
- torchtext
- Hugging Face's Tokenizers
- NumPy
- Matplotlib
- Seaborn
- TensorBoard

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/nmt.git
   cd nmt
2. **Create venv and install dependencies.py:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt