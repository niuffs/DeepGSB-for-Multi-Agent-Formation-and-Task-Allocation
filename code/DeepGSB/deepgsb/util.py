import logging
from typing import TYPE_CHECKING, Iterable, Tuple

import torch
from easydict import EasyDict as edict

if TYPE_CHECKING:
    from deepgsb.deepgsb import DeepGSB
    from options import Options

log = logging.getLogger(__file__)


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_stage(opt, stage):
    match = lambda freq: (freq > 0 and stage % freq == 0)
    return [match(opt.snapshot_freq), match(opt.ckpt_freq)]


def get_time(sec: float) -> Tuple[int, int, float]:
    h = int(sec // 3600)
    m = int((sec // 60) % 60)
    s = sec % 60
    return h, m, s


def flatten_dim01(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *x.shape[2:])


def unflatten_dim01(x: torch.Tensor, dim01) -> torch.Tensor:
    return x.reshape(*dim01, *x.shape[1:])


def save_checkpoint(opt: "Options", runner: "DeepGSB", keys: Iterable[str], stage_it: int) -> None:
    """Save intermediate checkpoint (stored in model_path)."""
    checkpoint = {}
    fn = opt.model_path + "/stage_{0:04}.npz".format(stage_it)
    for k in keys:
        variable = getattr(runner, k)
        if hasattr(variable, "state_dict"):
            checkpoint[k] = variable.state_dict()
        else:
            checkpoint[k] = variable
    torch.save(checkpoint, fn)
    log.info("Checkpoint saved: {}".format(fn))


def restore_checkpoint(opt: "Options", runner: "DeepGSB", load_name: str) -> None:
    assert load_name is not None
    log.info("Loading checkpoint {}...".format(load_name))
    full_keys = ['z_f', 'optimizer_f', 'z_b', 'optimizer_b']
    checkpoint = torch.load(load_name, map_location=opt.device)
    ckpt_keys = list(checkpoint.keys())
    for k in ckpt_keys:
        thing = getattr(runner, k)
        if hasattr(thing, "load_state_dict"):
            getattr(runner, k).load_state_dict(checkpoint[k])
        else:
            setattr(runner, k, checkpoint[k])

    if len(full_keys) != len(ckpt_keys):
        missing_keys = {k for k in set(full_keys) - set(ckpt_keys)}
        extra_keys = {k for k in set(ckpt_keys) - set(full_keys)}
        if len(missing_keys) > 0:
            log.warning("Does not load model for {}".format(missing_keys))
        if len(extra_keys) > 0:
            log.warning("Loaded extra keys: {}".format(extra_keys))
    else:
        log.info("Successfully loaded all modules")


def get_loss_str(loss) -> str:
    """Format a loss edict for log output.

    Renames keys for display (boundary -> init, kl -> end).
    Skips items whose absolute value is < 1e-8 (e.g. mid when there is
    no midpoint constraint, or grad when not actor-critic).
    """
    display_names = {
        'boundary': 'init',
        'mid': 'mid-kl',
        'kl': 'kl',
        'grad': 'grad',
        'bsde_td': 'bsde_td',
    }
    if isinstance(loss, edict):
        parts = []
        for key, val in loss.items():
            v = val.item() if hasattr(val, 'item') else float(val)
            if abs(v) < 1e-8:
                continue
            name = display_names.get(key, key)
            parts.append(f'({name}){v:+2.3f}')
        return ' '.join(parts)
    else:
        return f'{loss.item():+.4f}'
