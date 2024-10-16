from pathlib import Path

import torch

from pmdd.mindiffusion.ddpm import DDPM
from pmdd.mindiffusion.unet import NaiveUnet
from pmdd.utils.calc_utils import curl_2d, div_2d
from pmdd.utils.plot_utils import plot_ddpm_sample


def eval_ddpm(dim: int, res: int, n_samples: int, model_name: str, device: str) -> None:
    # Load the pre-trained model
    outpath = Path.cwd() / "output"
    params = torch.load(outpath / model_name)
    tot_curl = 0
    tot_div = 0
    tot_std = 0

    cfg = {
        "betas": (1e-4, 0.02),
        "n_T": 1000,
        "features": 16,
        "lr": 2e-4,
    }

    ddpm = DDPM(
        eps_model=NaiveUnet(dim, dim, cfg["features"]),
        betas=cfg["betas"],
        n_T=cfg["n_T"],
    )
    ddpm.to(device)

    # Load the state dictionary into the model
    ddpm.load_state_dict(params)

    ddpm.eval()
    with torch.no_grad():
        xh = ddpm.sample(n_samples, (dim, res, res), device)

        for sam in xh:
            tot_curl += abs(curl_2d(sam.cpu().numpy())).mean()
            tot_div += abs(div_2d(sam.cpu().numpy())).mean()
            tot_std += sam.std().item()

        _ = plot_ddpm_sample(xh.cpu(), save=True)

        print(f"Curl: {tot_curl / n_samples}")
        print(f"Div: {tot_div / n_samples}")
        print(f"Std: {tot_std / n_samples}")


if __name__ == "__main__":
    eval_ddpm(dim=2, res=32, n_samples=10, model_name="ddpm_test.pth", device="cuda:0")
