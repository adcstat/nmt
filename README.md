# NMT - Neural Machine Translation with Transformers

This project offers a framework for training, evaluating, and analyzing neural machine translation models using the Transformer architecture, with support for PyTorch's distributed training. It features dynamic parameter configuration, custom learning rate scheduling, attention weight visualization, and BLEU score computation for performance evaluation.

## Features

- **BPE Tokenization:** Efficient text tokenization with `bpe_tokenizer_train.py` supporting Byte Pair Encoding.
- **Dynamic Configuration:** JSON configuration files allow for flexible parameter adjustments.
- **Learning Rate Scheduler:** A custom scheduler supports optimal training with warmup and decay phases.
- **Distributed Training:** Leverage multiple GPUs for efficient training of Transformer models.
- **Checkpoint Averaging:** average over last checkpoints with `avg_checkpoints.py`.
- **Attention Visualization:** Visualize model attention mechanisms with `plot_entropies.py`.
- **BLEU Score Computation:** Quantitatively evaluate translation quality using `bleu.sh`.

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/nmt.git
   cd nmt
2. **Create venv and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
3. **Get data and train bpe tokenizer:**
    ```bash
    python bpe_tokenizer.py -v 32000 -ml 200
4. **Train a model:**
    ```bash
    torchrun --standalone --nproc_per_node=gpu train.py -v 32000 -pc base -mc plain_vanilla
5. **Average checkpoints:**
    ```bash
    python avg_checkpoints.py --path checkpoints/32000/plain_vanilla/base --last_n 5
6. **Plot entropies:**
    ```bash
    python plot_entropies.py -v 32000 -pc base -mc plain_vanilla -c averaged_weights_last_5
7. **Get bleu scores:**
    ```bash
    ./bleu.sh 32000 base plain_vanilla averaged_weights_last_5