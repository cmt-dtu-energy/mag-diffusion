from pathlib import Path

import torch
from torch.utils.data import DataLoader

import wandb
from pmdd.data import MagnetismData2D
from pmdd.loss import VPLoss
from pmdd.networks import VPPrecond
from pmdd.sample import sample
from pmdd.utils.calc_utils import curl_2d, div_2d
from pmdd.utils.plot_utils import plot_ddpm_sample


def train(device="cuda:0", wandb_=True) -> None:
    # start_time = time.time()
    datapath = Path.cwd() / "data"
    outpath = Path.cwd() / "output"
    n_samples = 1
    print_every = 100

    cfg = {
        "seed": 0,
        "epochs": 5000,
        # "betas": (1e-4, 0.02),
        # "n_T": 1000,
        # "features": 16,
        "lr": 5e-4,
        "batch_size": 256,
        "dim": 2,
        "res": 64,
        "max": False,
        "db_name": "magfield_symm_64_10000.h5",
    }
    if wandb_:
        wandb.init(entity="dl4mag", project="mag-diffusion-test", config=cfg)

    ddpm = VPPrecond(cfg["res"], cfg["dim"])
    ddpm.train().requires_grad_(True).to(device)
    dataloader = DataLoader(
        MagnetismData2D(datapath, cfg["db_name"], cfg["max"], norm_=False),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=15,
    )
    optim = torch.optim.Adam(ddpm.parameters(), lr=cfg["lr"])

    # variance-preserving (vp)
    loss_fn = VPLoss()

    for i in range(cfg["epochs"]):
        ddpm.train()

        loss_ema = None
        for x in dataloader:
            optim.zero_grad()
            x = x.to(device)  # noqa: PLW2901
            loss = loss_fn(ddpm, x).sum()
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            optim.step()

        if i % print_every == 0:
            # ddpm.eval()
            # with torch.no_grad():
            xh = sample(
                ddpm,
                n_samples,
                device,
                num_steps=200,
                sigma=[0.002, 80],
                rho=7,
                zeta_pde=0,
            )
            tot_curl = 0
            tot_div = 0
            tot_std = 0

            for sam in xh:
                sam = sam.detach().cpu()  # noqa: PLW2901
                tot_curl += abs(curl_2d(sam)).mean()
                tot_div += abs(div_2d(sam)).mean()
                tot_std += sam.std()

            fig = plot_ddpm_sample(xh.detach().cpu())
            if wandb_:
                wandb.log(
                    {
                        "loss": loss_ema,
                        "curl": tot_curl / n_samples,
                        "div": tot_div / n_samples,
                        # "std": tot_std / n_samples,
                        "sample": wandb.Image(fig),
                    },
                    step=i,
                )

                print(loss_ema)

        torch.save(ddpm.state_dict(), outpath / "ddpm_test_pde.pth")

    if wandb_:
        wandb.finish()


if __name__ == "__main__":
    train(wandb_=True)
