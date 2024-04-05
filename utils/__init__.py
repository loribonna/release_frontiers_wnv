import json
import os
import re
import torch


def load_checkpoint_fix_statedict(model, state_dict, params, args):
    if args.mode != 'graph':
        for k in list(state_dict.keys()):
            if "conv1" in k:
                del state_dict[k]
    for k in list(state_dict.keys()):
        if "conv_1" in k:
            del state_dict[k]
    if args.mode != 'graph':
        for k in list(state_dict.keys()):
            if 'layer' in k:
                new_k = re.sub(
                    r"(?P<p>.*)(?P<s>layer)(?P<n>\w)(?P<e>.*)", r"\1\2\3.blocks\4", k)
                state_dict[new_k] = state_dict.pop(k)
        for k in list(state_dict.keys()):
            # for cond bn
            if "bn" in k:
                del state_dict[k]

    missing, _ = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        ks = []
        for k in missing:
            if k.startswith("model") and "downsample" not in k and "conv_1" not in k:
                if "conv1" in k:
                    if args.mode == 'graph':
                        ks.append(k)
                elif "bn" in k:
                    if args.mode == 'graph':
                        ks.append(k)
                else:
                    ks.append(k)

        assert len(ks) == 0, f"Missing keys: {ks}"

    return model


def get_data_dir():
    """
    Check if is_slurm and return data path.
    """
    # dataset_name = '19TO22'
    dataset_name = 'frontiers_wnv_data'
    data_dir = os.path.join(os.getcwd(), 'data', dataset_name)

    print(f"Data dir: {data_dir}")
    return data_dir


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_dict_to_json_append(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'a') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, checkpoint, split=0):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, f'model{split}.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
