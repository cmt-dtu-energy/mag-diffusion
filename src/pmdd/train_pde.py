from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import wandb
from pmdd.data import MagnetismData2D
from pmdd.loss import EDMLoss, VPLoss
from pmdd.networks import EDMPrecond, VPPrecond
from pmdd.sample import sample
from pmdd.utils.calc_utils import curl_2d, div_2d
from pmdd.utils.plot_utils import plot_ddpm_sample


def train(cfg, device="cuda:1", wandb_=True, save_name = "") -> None:
    start_t = datetime.now()
    #datapath = Path.cwd() / "data"
    datapath = "/home/s214435/data"
    outpath = Path.cwd() / "output"
    n_samples = 1
    print_every = 100
    save_every = 1000

    if wandb_:
        wandb.init(entity="dl4mag", project="mag-diffusion", config=cfg)

    torch.manual_seed(cfg["seed"])
    ddpm = EDMPrecond(cfg["res"], cfg["dim"], model_channels = cfg["model_channels"], channel_mult = cfg["channel_mult"], num_blocks = cfg["num_blocks"], sigma_data=1)
    ddpm.train().requires_grad_(True).to(device)
    dataloader = DataLoader(
        MagnetismData2D(datapath, cfg["db_name"], cfg["max"], norm_=False, max_observations=cfg["data_limit"]),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=15,
    )
    optim = torch.optim.Adam(ddpm.parameters(), lr=cfg["lr"])
    loss_fn = EDMLoss(sigma_data=1)
    cur_nimg = 0

    for i in range(cfg["epochs"]+1):
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

            # Update weights.
            for g in optim.param_groups:
                g["lr"] = cfg["lr"] * min(cur_nimg / 1e7, 1)
            for param in ddpm.parameters():
                if param.grad is not None:
                    torch.nan_to_num(
                        param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                    )
            optim.step()

            cur_nimg += cfg["batch_size"]

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
                        "sample": wandb.Image(fig)
                    },
                    step=i,
                )

                print(loss_ema)

        if i % save_every == 0:
            torch.save(
                ddpm.state_dict(), f"ddpm_{save_name}_epoch_{i}.pth"
                #{start_t.strftime('%m-%d_%H-%M')}_
            )

    if wandb_:
        wandb.finish()


if __name__ == "__main__":

    #channel_mult = [1,2,3,4]
    #num_blocks = [1,2,3]

    # for i in range(3):
    #     cfg = {
    #         "seed": 0,
    #         "epochs": 3000,
    #         "lr": 1e-3,
    #         "batch_size": 500,
    #         "dim": 2,
    #         "res": 64,
    #         "max": False,
    #         "db_name": "magfield_symm_64_30000.h5",
    #         "model_channels" : 16,
    #         "channel_mult" : channel_mult[0:i+2],
    #         "num_blocks" : num_blocks[i]
    #     }
    #    train(cfg, wandb_=True, save_name=f"layers_{len(channel_mult[0:i+2])}_blocks_{num_blocks[i]}")

    model_channels = [4,8, 16]

    for i in range(3):
        cfg = {
            "seed": 0,
            "epochs": 3000,
            "lr": 1e-3,
            "batch_size": 500,
            "dim": 2,
            "res": 64,
            "max": False,
            "db_name": "magfield_symm_64_30000.h5",
            "model_channels" : model_channels[i],
            "channel_mult" : [1,2],
            "num_blocks" : 1,
            "data_limit" : None
        }
        train(cfg, wandb_=True, save_name=f"small_channels_{model_channels[i]}")

    # cfg_data = {
    #     "seed": 0,
    #     "epochs": 2000,
    #     "lr": 1e-3,
    #     "batch_size": 500,
    #     "dim": 2,
    #     "res": 64,
    #     "max": False,
    #     "db_name": "magfield_symm_64_30000.h5",
    #     "model_channels" : 16,
    #     "channel_mult" : [1,2,3],
    #     "num_blocks" : 2,
    #     "data_limit" : 10000
    # }
    # train(cfg_data, wandb_=True, save_name=f"data_limit_{10000}_epochs_{20000}")

