from pathlib import Path

import torch

import s3prl
from s3prl.upstream.utils import load_fairseq_ckpt, merge_with_parent
from s3prl.upstream.wav2vec2.wav2vec2_model import (
    AudioPretrainingConfig,
    Wav2Vec2Config,
    Wav2Vec2Model,
)
from s3prl.upstream.wav2vec2.wav2vec2_cond_model import Wav2Vec2CondModel

def load_and_convert_fairseq_ckpt(fairseq_source: str, output_path: str = None):
    state, cfg = load_fairseq_ckpt(fairseq_source)
    output_state = {
        "task_cfg": cfg["task"],
        "model_cfg": cfg["model"],
        "model_weight": state["model"],
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(output_state, output_path)


def load_converted_model(ckpt: str):
    ckpt_state = torch.load(ckpt, map_location="cpu")

    for required_key in ["task_cfg", "model_cfg", "model_weight"]:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt} is not a valid checkpoint since the required key: {required_key} is missing"
            )

    task_cfg = merge_with_parent(AudioPretrainingConfig, ckpt_state["task_cfg"])
    model_cfg = merge_with_parent(Wav2Vec2Config, ckpt_state["model_cfg"])
    model = Wav2Vec2Model(model_cfg)
    model.load_state_dict(ckpt_state["model_weight"])
    return model, task_cfg




def load_condition_converted_model(ckpt: str, **kwargs):
    ckpt_state = torch.load(ckpt, map_location="cpu")

    for required_key in ["task_cfg", "model_cfg", "model_weight"]:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt} is not a valid checkpoint since the required key: {required_key} is missing"
            )

    task_cfg = merge_with_parent(AudioPretrainingConfig, ckpt_state["task_cfg"])
    model_cfg = merge_with_parent(Wav2Vec2Config, ckpt_state["model_cfg"])
    model = Wav2Vec2Model(model_cfg)
    model.load_state_dict(ckpt_state["model_weight"])

    cond_cfg = kwargs
    condition_model = Wav2Vec2CondModel(model_cfg, cond_cfg)
    copy_init_condition_weights(model, condition_model)

    return condition_model, task_cfg, cond_cfg

def copy_init_condition_weights(model, condition_model):
    for name, param in condition_model.named_parameters():
        if name in model.state_dict():
            param.data.copy_(model.state_dict()[name].data)
        
        if "linear_scale.weight" in name:
            param.data = torch.zeros_like(param)
        elif "linear_scale.bias" in name:
            param.data = torch.ones_like(param)
        elif "linear_shift.weight" in name or "linear_shift.bias" in name:
            param.data = torch.zeros_like(param)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fairseq_ckpt")
    parser.add_argument("--output_name")
    parser.add_argument(
        "--output_dir", default=Path(s3prl.__file__).parent.parent / "converted_ckpts"
    )
    args = parser.parse_args()

    Path(args.output_dir).parent.mkdir(exist_ok=True, parents=True)
    if args.output_name is not None:
        stem = args.output_name
    else:
        stem = Path(args.fairseq_ckpt).stem

    load_and_convert_fairseq_ckpt(
        args.fairseq_ckpt, Path(args.output_dir) / f"{stem}.pt"
    )
