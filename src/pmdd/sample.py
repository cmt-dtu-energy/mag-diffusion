import itertools

import torch
import tqdm


def sample(
    net, n_samples, device, num_steps, sigma, rho, zeta_pde, div_loss=False
) -> None:
    print(f"Generating {n_samples} samples...")
    latents = torch.randn(
        [n_samples, net.img_channels, net.img_resolution, net.img_resolution],
        device=device,
    )

    sigma_min = max(sigma[0], net.sigma_min)
    sigma_max = min(sigma[1], net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    sigma_t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_t_steps = torch.cat(
        [net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]
    )  # t_N = 0

    x_next = latents.to(torch.float64) * sigma_t_steps[0]
    # TODO Random index for observations

    ############################ Sample the data ############################
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(
        list(enumerate(itertools.pairwise(sigma_t_steps))),
        unit="step",
    ):  # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)

        # Euler step
        x_N = net(x_cur, sigma_t).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

        # 2nd order correction
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

        # TODO Scale the data back

        # Div 2D
        if div_loss:
            Fx_x = torch.gradient(x_N[:, 0:1], dim=2)[0]
            Fy_y = torch.gradient(x_N[:, 1:2], dim=3)[0]
            div = torch.cat([Fx_x, Fy_y], dim=1).sum(dim=1)
            L_pde = div.abs().mean()
            grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]

        # if obs_loss:
        #     obs_loss = (u - u_GT).squeeze()
        #     L_obs_u = (obs_loss * u_mask)**2

        #     grad_x_cur_obs_u = torch.autograd.grad(
        #         outputs=L_obs_u, inputs=x_cur, retain_graph=True
        #     )[0]

        # if i <= 0.8 * num_steps:
        #     x_next = x_next  # - zeta_obs_u * grad_x_cur_obs_u
        # else:

        if i > 0.8 * num_steps and div_loss:
            x_next = (
                x_next
                # - 0.1 * (zeta_obs_u * grad_x_cur_obs_u)
                - zeta_pde * grad_x_cur_pde
            )

    return x_N
