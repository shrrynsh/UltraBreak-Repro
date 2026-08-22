"""
optimise_proj.py
----------------
Projection-enabled variant of optimise.py.

The released trainer omits the Section 3.2 "Projection onto Constrained Input
Space": project_patch() (utils.py) is dead code, so the raw patch is fed to the
model. This script wires that projection in, exactly as the paper describes:

    x_proj = clip(gamma * x + beta, 0, 1),   gamma = CLIP_STD, beta = CLIP_MEAN

and feeds x_proj (the projected patch) into the patch-application operator A(),
matching Eq. 10 A(x_blank, x_proj, l, r, s). The optimiser still updates the raw
latent x; only what the model sees — and what gets saved as the deployable
image — is the projected patch. The TV loss stays on the raw x, per Eq. 10
L_TV(x). Everything else (Adam, lr=0.01, transforms, corpus, 224x224) is
unchanged from optimise.py so this isolates the projection's effect.

Two flags control how faithfully the projection is applied:

  --clamp_latent   also clamp the raw latent to [0,1] after each step. The
                   paper applies exactly one constraint and it sits INSIDE the
                   projection, so this extra clamp makes the paper's own clip
                   unreachable. Off by default; pass it to reproduce job 625.
  --init_space     'unit'  = x ~ U(0,1) in latent space (what optimise.py did,
                            back when x was the image itself)
                   'image' = uniform random image mapped back through the
                            projection, x = (p - beta) / gamma

See repro_notes/DISCREPANCIES.md (D1, O1, O2) for the reasoning.
"""

from PIL import Image

from llava16_adapter import Llava16Adapter
from qwen2_adapter import Qwen2Adapter
from utils import (
    AblationConfig,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    project_patch,
    set_seed,
)

import torch.optim as optim

import torch
import torchvision.transforms as transforms
import pandas as pd
import time
import random

import os
import json
import pickle


def initialise_patch(device, patch_size, image_path=None, init_space="unit"):
    """
    init_space controls what "random initialisation" means once the projection
    is active and adv_patch is a latent rather than the image itself.

      "unit"  - x ~ U(0,1) in latent space. This is what optimise.py did, where
                x WAS the image. Under the projection it starts the run inside
                the narrow band gamma*[0,1]+beta = [0.41, 0.75], i.e. a washed
                out mid-grey square with ~3.5x too little contrast.
      "image" - draw a uniform random *image* in [0,1] and map it back through
                the projection, x = (p - beta) / gamma, so the starting image
                spans the full range as it did before the projection existed.
    """
    mean = torch.tensor(OPENAI_CLIP_MEAN, device=device).view(-1, 1, 1)
    std = torch.tensor(OPENAI_CLIP_STD, device=device).view(-1, 1, 1)

    if image_path is not None:
        transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor()
        ])
        img = Image.open(image_path).convert("RGB")
        adv_patch = transform(img).to(device)
        if init_space == "image":
            # Checkpoints are saved PROJECTED, so invert the projection to
            # recover the latent the optimiser was actually updating.
            adv_patch = (adv_patch - mean) / std
    elif init_space == "image":
        adv_patch = (torch.rand(3, patch_size, patch_size, device=device) - mean) / std
    else:
        adv_patch = torch.rand(3, patch_size, patch_size, device=device)

    adv_patch = adv_patch.detach().clone()
    adv_patch.requires_grad_()
    return adv_patch


def save_tensor_as_image(tensor: torch.Tensor, save_path: str) -> None:
    """Convert a [C,H,W] or [1,C,H,W] tensor in [0,1] to a PIL image and save it."""
    tensor = tensor.cpu().clone().detach()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = (tensor * 255).to(torch.uint8)
    np_img = tensor.permute(1, 2, 0).numpy()
    Image.fromarray(np_img).save(save_path)


def total_variation(img):
    """TV loss for [C,H,W] or [B,C,H,W]."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    h_variation = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    w_variation = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return h_variation + w_variation


def project_to_constrained_space(adv_patch, gamma, beta):
    """
    Section 3.2 projection: x_proj = clip(gamma * x + beta, 0, 1), per-channel
    gamma = CLIP_STD, beta = CLIP_MEAN. Differentiable, so gradients flow to the
    raw latent adv_patch.
    """
    projected = project_patch(adv_patch, scale=gamma, shift=beta)
    return torch.clamp(projected, 0.0, 1.0)


SURROGATES = {
    "qwen": ("Qwen/Qwen2-VL-7B-Instruct", Qwen2Adapter),
    "qwen2b": ("Qwen/Qwen2-VL-2B-Instruct", Qwen2Adapter),   # Fig 2a (C2a) small surrogate
    "llava": ("llava-hf/llava-v1.6-mistral-7b-hf", Llava16Adapter),
}


def main(args=None):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    train_config = args.train_config if args else "./train_configs/safebench-tiny_jailbroken_mode.csv"

    exp_name = args.exp_name if args else 'safebench-tiny_jailbroken_mode_proj_5000'
    base_epoch = args.base_epoch if args else 0
    init_space = args.init_space if args else "unit"
    projection = not args.no_projection
    # Without the projection the latent IS the image, so the released code's
    # clamp(x, 0, 1) is the only thing keeping it in range and must stay on.
    clamp_latent = args.clamp_latent or not projection
    lr = args.lr
    num_epochs = args.num_epochs if args else 5000
    tv_weight = args.tv_weight
    l2_weight = args.l2_weight
    custom_loss = args.loss == "semantic"

    if not projection and init_space == "image":
        raise SystemExit("--init_space image is meaningless without the projection; use 'unit'.")

    # Seed before anything samples: patch init, transform schedule, embedding noise.
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[cfg] seed={args.seed}")
    else:
        print("[cfg] WARNING: no --seed given; this run is not reproducible and "
              "cannot contribute to a variance study.")

    cfg = AblationConfig(
        loss=args.loss,
        loss_mode=args.loss_mode,
        tau=args.tau,
        embed_noise=args.embed_noise,
        pos_alpha=args.pos_alpha,
        transforms=not args.no_transforms,
        normalise_patch=not args.raw_patch_injection,
    )

    surrogate_names = [s.strip() for s in args.surrogates.split(",") if s.strip()]
    unknown = [s for s in surrogate_names if s not in SURROGATES]
    if unknown:
        raise SystemExit(f"Unknown surrogate(s) {unknown}; choose from {sorted(SURROGATES)}")
    ensemble = []
    for name in surrogate_names:
        model_id, adapter_cls = SURROGATES[name]
        ensemble.append(adapter_cls(model_id, patch_only=False, cfg=cfg))

    os.makedirs(f"outputs/{exp_name}/", exist_ok=True)
    target_df = pd.read_csv(train_config)
    target_df = target_df.fillna("")

    if base_epoch > 0:
        adv_patch = initialise_patch(device, 224, f'outputs/{exp_name}/{base_epoch}.png', init_space)
    else:
        adv_patch = initialise_patch(device, 224, None, init_space)

    # Fixed per-channel projection parameters (CLIP normalisation statistics).
    gamma = torch.tensor(OPENAI_CLIP_STD, device=device)
    beta = torch.tensor(OPENAI_CLIP_MEAN, device=device)
    if projection:
        print(f"[proj] projection active: x_proj = clip(CLIP_STD * x + CLIP_MEAN, 0, 1)")
        print(f"[proj] gamma(CLIP_STD)={OPENAI_CLIP_STD}  beta(CLIP_MEAN)={OPENAI_CLIP_MEAN}")
    else:
        print("[proj] projection DISABLED — reproducing the released code's path "
              "(raw patch, clamp(x, 0, 1) only)")
    print(f"[proj] clamp_latent={clamp_latent}  init_space={init_space}")

    def project(x):
        """The deployable image for the current latent."""
        return project_to_constrained_space(x, gamma, beta) if projection else x

    with torch.no_grad():
        _img0 = project(adv_patch)
        print(f"[proj] init latent range [{adv_patch.min():+.3f}, {adv_patch.max():+.3f}]  "
              f"-> image [{_img0.min():.3f}, {_img0.max():.3f}] std={_img0.std():.4f}")

    # Record exactly what this run was, next to its outputs.
    resolved = {
        "exp_name": exp_name,
        "train_config": train_config,
        "seed": args.seed,
        "surrogates": surrogate_names,
        "projection": projection,
        "clamp_latent": clamp_latent,
        "init_space": init_space,
        "lr": lr,
        "num_epochs": num_epochs,
        "base_epoch": base_epoch,
        "tv_weight": tv_weight,
        "l2_weight": l2_weight,
        "ablation": cfg.as_dict(),
        "torch": torch.__version__,
    }
    with open(f"outputs/{exp_name}/run_config.json", "w") as fh:
        json.dump(resolved, fh, indent=2)
    print(f"[cfg] {json.dumps(resolved)}")

    optimizer = optim.Adam([adv_patch], lr=lr)

    patience = 5000
    best_loss = float('inf')
    best_img = None
    no_improve_count = 0

    losses = []

    for epoch in range(base_epoch, num_epochs):
        start_time = time.time()
        optimizer.zero_grad()
        text_loss = 0
        target_losses = []
        for model in ensemble:
            for index, row in target_df.iterrows():
                print_probs = (index == 0 and epoch % 20 == 0)

                # TV / L2 regularisers act on the raw latent x (Eq. 10 L_TV(x)).
                l2_loss = torch.mean((adv_patch - 0.5) ** 2)
                tv_loss = total_variation(adv_patch)

                # Project before the patch-application operator: model sees x_proj.
                proj_patch = project(adv_patch)

                model_loss = model.compute_loss(row, proj_patch, '', custom_loss=custom_loss, print_probs=print_probs)
                loss = model_loss + tv_loss * tv_weight + l2_loss * l2_weight
                loss.backward()
                text_loss += model_loss.item()
                target_losses.append(model_loss.item())

        total_loss = text_loss / (len(target_df) * len(ensemble)) + tv_loss * tv_weight + l2_loss * l2_weight

        optimizer.step()

        # Section 3.2 applies exactly ONE constraint, and it lives inside the
        # projection: x_proj = clip(gamma*x + beta, 0, 1). Clamping the raw
        # latent as well makes that clip unreachable, since gamma*[0,1]+beta is
        # a strict subset of [0,1] -- so the paper's clip can never bind and the
        # deployable image is squeezed into ~27% of each channel's range.
        # Off by default; --clamp_latent restores the old (job 625) behaviour.
        if clamp_latent:
            adv_patch.data = torch.clamp(adv_patch.data, 0, 1)

        if epoch % 20 == 0:
            with torch.no_grad():
                print("saving..")
                save_path = f"outputs/{exp_name}/{epoch}.png"
                # Save the PROJECTED patch — that is the deployable image and
                # what inference must use for consistency with training.
                save_tensor_as_image(project(adv_patch), save_path)

                for model in ensemble:
                    print(model.generate(target_df.iloc[0]['text'], save_path))

            df = pd.DataFrame(losses)
            df.to_csv(f"outputs/{exp_name}/losses.csv", index=False)

        if total_loss < best_loss:
            best_loss = total_loss
            best_img = adv_patch.clone().detach()
            no_improve_count = 0
        else:
            no_improve_count += 1

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch}: Avg Loss = {total_loss}, Text Loss = {text_loss/ (len(target_df) * len(ensemble))}, TV Loss = {tv_loss * tv_weight}, L2 Loss = {l2_loss * l2_weight}, Time Spent = {epoch_duration}s")

        losses.append({
            'epoch': epoch,
            'total_loss': total_loss.item(),
            'text_loss': text_loss,
            'tv_loss': tv_loss.item(),
            'tv_weight': tv_weight,
            'l2_loss': l2_loss.item(),
            'l2_weight': l2_weight,
        })

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}. Reverting to best model state.")
            adv_patch.data = best_img.data
            break

    # Final save is also the projected patch.
    save_tensor_as_image(project(adv_patch), f"outputs/{exp_name}/adv_patch_{epoch}.png")
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default="./train_configs/safebench-tiny_jailbroken_mode.csv")
    parser.add_argument("--exp_name", type=str, default="safebench-tiny_jailbroken_mode_proj_5000")
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--base_epoch", type=int, default=0)
    parser.add_argument("--clamp_latent", action="store_true",
                        help="Legacy: also clamp the raw latent to [0,1] after each step. "
                             "Makes the Section 3.2 clip unreachable. Reproduces job 625.")
    parser.add_argument("--init_space", type=str, default="unit", choices=["unit", "image"],
                        help="'unit': x ~ U(0,1) in latent space (legacy). "
                             "'image': uniform random image mapped back through the projection.")
    parser.add_argument("--no_projection", action="store_true",
                        help="Disable the Section 3.2 projection entirely, reproducing the "
                             "released code's path (raw patch + clamp(x,0,1)). Implies "
                             "--clamp_latent, since the latent is then the image itself.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for patch init, the transform schedule and embedding noise. "
                             "The released code seeds nothing; without this, run-to-run "
                             "variance is unquantified.")

    # Optimisation (Table 4 defaults)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tv_weight", type=float, default=0.5,
                        help="lambda_TV (Table 4: 0.5). Set 0 for the no-TV ablation cell.")
    parser.add_argument("--l2_weight", type=float, default=0.0,
                        help="Undocumented L2 pull toward mid-grey; inert at 0 (see "
                             "DISCREPANCIES.md D8). Its neutral point is 0.5 in pixel space "
                             "but 0.0 in latent space, so do not enable it with --projection.")

    # Surrogate (paper: "A single surrogate is sufficient")
    parser.add_argument("--surrogates", type=str, default="qwen",
                        help="Comma-separated surrogate list, e.g. 'qwen' or 'qwen,llava'. "
                             "More than one reproduces the ensemble the paper argues is "
                             "unnecessary.")

    # Semantic loss (paper: "Effect of Semantic Loss")
    parser.add_argument("--loss", type=str, default="semantic", choices=["semantic", "ce"],
                        help="'semantic' is Eq. 8; 'ce' is the cross-entropy baseline.")
    parser.add_argument("--loss_mode", type=str, default="attention", choices=["token", "attention"],
                        help="'token' is exact per-token matching (the tau->0 end); "
                             "'attention' is the relaxed objective the paper uses.")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="Attention temperature (Table 4: 0.5). Large values approximate "
                             "the tau->infinity cell; only meaningful with --loss_mode attention.")
    parser.add_argument("--embed_noise", type=float, default=1e-4, help="Table 4 epsilon.")
    parser.add_argument("--pos_alpha", type=float, default=0.01)

    # Input transformations (paper: "Effect of Transformation and Regularisation")
    parser.add_argument("--raw_patch_injection", action="store_true",
                        help="Reproduce the DISCREPANCIES D11 bug: inject the raw [0,1] patch "
                             "into the already-normalised pixel tensor (job 625's regime). "
                             "Default off = the fix (patch normalised into CLIP space).")
    parser.add_argument("--no_transforms", action="store_true",
                        help="Composite the patch unscaled, unrotated and centred — the "
                             "paper's 'no constraints' ablation cell.")

    main(parser.parse_args())
