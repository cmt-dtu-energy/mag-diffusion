from pathlib import Path

import numpy as np
import torch

from pmdd.mindiffusion.ddpm import DDPM, DDPMGuided
from pmdd.mindiffusion.unet import NaiveUnet
from pmdd.networks import VPPrecond
from pmdd.sample import sample
from pmdd.utils.calc_utils import curl_2d, div_2d
from pmdd.utils.plot_utils import plot_ddpm_sample


def eval_ddpm(
    dim: int,
    res: int,
    n_samples: int,
    model_name: str,
    device: str,
    div_loss=bool,
    guide_weight: int = 1,
) -> None:
    # For reproducibility
    torch.manual_seed(42)
    # Load the pre-trained model
    outpath = Path.cwd() / "output"
    params = torch.load(outpath / model_name, weights_only=True)
    tot_curl = 0
    tot_div = 0
    tot_std = 0

    cfg = {
        "betas": (1e-4, 0.02),
        "n_T": 1000,
        "features": 16,
    }

    if div_loss:
        ddpm = DDPMGuided(
            eps_model=NaiveUnet(dim, dim, cfg["features"]),
            betas=cfg["betas"],
            n_T=cfg["n_T"],
        )
    else:
        ddpm = DDPM(
            eps_model=NaiveUnet(dim, dim, cfg["features"]),
            betas=cfg["betas"],
            n_T=cfg["n_T"],
        )
    ddpm.to(device)
    ddpm.load_state_dict(params)

    if div_loss:
        xh = ddpm.sample(n_samples, (dim, res, res), device, guide_weight)
    else:
        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(n_samples, (dim, res, res), device)

    for sam in xh:
        sam = sam.detach().cpu()  # noqa: PLW2901
        tot_curl += np.abs(curl_2d(sam)).mean()
        tot_div += np.abs(div_2d(sam)).mean()
        tot_std += sam.std().item()

    _ = plot_ddpm_sample(
        xh.detach().cpu(), figname=f"ddpm_sample_{div_loss!s}", save=True
    )

    print(f"Curl: {tot_curl / n_samples}")
    print(f"Div: {tot_div / n_samples}")
    print(f"Std: {tot_std / n_samples}")


def eval_ddpm_pde(
    dim: int,
    res: int,
    n_samples: int,
    model_name: str,
    device: str,
    div_loss: bool = False,
    plot_div: bool = False,
) -> None:
    # For reproducibility
    torch.manual_seed(42)
    # Load the pre-trained model
    outpath = Path.cwd() / "output"
    params = torch.load(outpath / model_name, weights_only=True)
    tot_curl = 0
    tot_div = 0
    tot_std = 0

    ddpm = VPPrecond(res, dim)
    ddpm.train().requires_grad_(True).to(device)

    # Load the state dictionary into the model
    ddpm.load_state_dict(params)

    xh = sample(
        ddpm,
        n_samples,
        device,
        num_steps=200,
        sigma=[0.002, 80],
        rho=7,
        zeta_pde=10,
        div_loss=div_loss,
    )

    if plot_div:
        np_div = np.zeros((n_samples, res, res))

    for i, sam in enumerate(xh):
        sam = sam.detach().cpu()  # noqa: PLW2901
        tot_curl += np.abs(curl_2d(sam)).mean()
        tot_div += np.abs(div_2d(sam)).mean()
        tot_std += sam.std()

        if plot_div:
            np_div[i] = div_2d(sam)

    _ = plot_ddpm_sample(xh.detach().cpu(), figname=f"diff_pde_{div_loss!s}", save=True)

    if plot_div:
        _ = plot_ddpm_sample(
            np.expand_dims(np_div, axis=1),
            figname=f"div_pde_{div_loss!s}",
            vmax=1e-3,
            save=True,
        )

    print(f"Curl: {tot_curl / n_samples}")
    print(f"Div: {tot_div / n_samples}")
    print(f"Std: {tot_std / n_samples}")


if __name__ == "__main__":
    # eval_ddpm_pde(
    #     dim=2,
    #     res=64,
    #     n_samples=3,
    #     model_name="ddpm_test_pde.pth",
    #     device="cuda:0",
    #     div_loss=True,
    #     plot_div=True,
    # )

    # dim: int, res: int, n_samples: int, model_name: str, device: str
    eval_ddpm(
        dim=2,
        res=64,
        n_samples=2,
        model_name="ddpm_test_max.pth",
        device="cuda:0",
        div_loss=True,
        guide_weight=10,
    )
