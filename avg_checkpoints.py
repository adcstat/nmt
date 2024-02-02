import argparse
import os
import copy
import torch


def average_checkpoints(cp_dir_path, last_n):
    """
    Average the parameters of the checkpoints.
    """
    cp_paths = [os.path.join(cp_dir_path, fn) for fn in os.listdir(cp_dir_path) if fn.startswith("checkpoint_")]
    cp_paths.sort()
    cp_paths = cp_paths[-last_n:]
    
    # Load the first checkpoint
    avg_checkpoint = torch.load(cp_paths[0])["MODEL_STATE"]
    print(f"processed {cp_paths[0]}")

    # Initialize a dictionary to store the averaged weights
    averaged_weights = copy.deepcopy(avg_checkpoint)
    
    # Iterate over remaining checkpoints and accumulate the weights
    for path in cp_paths[1:]:
        checkpoint = torch.load(path)["MODEL_STATE"]
        for key in averaged_weights.keys():
            averaged_weights[key] += checkpoint[key]
        print(f"processed {path}")
    
    # Average the weights
    for key in averaged_weights.keys():
        averaged_weights[key] /= len(cp_paths)
    
    torch.save({"MODEL_STATE": averaged_weights}, f"{cp_dir_path}/averaged_weights_last_{last_n}.tar")
    print("saved averaged checkpoint")


def main():
    parser = argparse.ArgumentParser(description="Load checkpoints and save loss arrays.")
    parser.add_argument("--path", required=True, type=str, help="Path to the checkpoint directory")
    parser.add_argument("--last_n", required=True, type=int, help="How many of the last checkpoints to average over")
    args = parser.parse_args()
    average_checkpoints(args.path, args.last_n)

if __name__ == "__main__":
    main()


