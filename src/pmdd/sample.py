import itertools

import torch
import tqdm


def sample(
    net, n_samples, device, num_steps, sigma, rho, pde_loss=False, zeta_pde=100, obs_loss=False, zeta_obs = 100, observations = None, 
    mask = None, obs_scaling = 1, init_noise=None
) -> None:
    print(f"Generating {n_samples} samples...")
    samples = []
    latents = init_noise
    if init_noise == None:
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

    ############################ Sample the data ############################
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(
        list(enumerate(itertools.pairwise(sigma_t_steps))),
        unit="step",
    ):  # 0, ..., N-1
        x_cur = x_next.detach().clone()
        if i % 10 == 0:
            samples.append(x_cur)
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

        x_N_scaled = x_N*obs_scaling

        # Div 2D
        if pde_loss:
            Fx_x = torch.gradient(x_N_scaled[:, 0:1], dim=2)[0]
            Fy_y = torch.gradient(x_N_scaled[:, 1:2], dim=3)[0]
            div = torch.cat([Fx_x, Fy_y], dim=1).sum(dim=1)
            L_div = torch.linalg.norm(div)/(net.img_resolution**2)
            Fx_y = torch.gradient(x_N_scaled[:, 0:1], dim=3)[0]
            Fy_x = torch.gradient(x_N_scaled[:, 1:2], dim=2)[0]
            curl = Fy_x-Fx_y
            L_curl = torch.linalg.norm(curl)/(net.img_resolution**2)
            L_pde = L_div + L_curl
            grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur, retain_graph=True)[0]
    
        if obs_loss:
            L_obs = torch.linalg.norm((x_N_scaled-observations*mask)*mask)/torch.sum(mask)

            grad_x_cur_obs = torch.autograd.grad(
                outputs=L_obs, inputs=x_cur, retain_graph=True
            )[0]

        if i <= 0.8 * num_steps and obs_loss:
            x_next = (
                    x_next 
                    - zeta_obs * grad_x_cur_obs
            )

        if i > 0.8 * num_steps:
            if pde_loss:
                x_next = (
                    x_next
                    - zeta_pde * grad_x_cur_pde
                )
            if obs_loss:
                x_next = (
                    x_next
                    - 0.1 * (zeta_obs * grad_x_cur_obs)
                )

    return x_next*obs_scaling, samples
