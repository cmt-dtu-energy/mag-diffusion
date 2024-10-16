import itertools

import torch
import torch.nn.functional as F
import tqdm

# def random_index(k, grid_size, seed=0, device=torch.device("cuda")) -> torch.Tensor:
#     """randomly select k indices from a [grid_size, grid_size] grid."""
#     rng = np.random.default_rng(seed)
#     indices = rng.choice(grid_size**2, k, replace=False)
#     indices_2d = np.unravel_index(indices, (grid_size, grid_size))
#     indices_list = list(zip(indices_2d[0], indices_2d[1], strict=False))
#     mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
#     for i in indices_list:
#         mask[i] = 1
#     return mask


# def get_obs_loss(u, u_GT, u_mask, device=torch.device("cuda")) -> tuple[torch.Tensor]:
#     """Return the loss of the Darcy Flow equation and the observation loss."""
#     deriv_x = (
#         torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(1, 1, 1, 3)
#         / 2
#     )
#     deriv_y = (
#         torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(
#             1, 1, 3, 1
#         )
#         / 2
#     )
#     grad_x_next_x = F.conv2d(u, deriv_x, padding=(0, 1))
#     grad_x_next_y = F.conv2d(u, deriv_y, padding=(1, 0))
#     result = F.conv2d(grad_x_next_x, deriv_x, padding=(0, 1)) + F.conv2d(
#         grad_x_next_y, deriv_y, padding=(1, 0)
#     )
#     pde_loss = result + 1
#     pde_loss = pde_loss.squeeze()

#     observation_loss_u = (u - u_GT).squeeze()
#     observation_loss_u = observation_loss_u * u_mask

#     return pde_loss, observation_loss_u


def sample(net, n_samples, device, num_steps, sigma, rho, zeta_pde) -> None:
    ############################ Set up EDM latent ############################
    print(f"Generating {n_samples} samples...")
    latents = torch.randn(
        [n_samples, net.img_channels, net.img_resolution, net.img_resolution],
        device=device,
    )
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[
            torch.randint(net.label_dim, size=[n_samples], device=device)
        ]

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
    # known_index_a = random_index(500, 128, seed=1)
    # known_index_u = random_index(500, 128, seed=0)

    ############################ Sample the data ############################
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(
        list(enumerate(itertools.pairwise(sigma_t_steps))),
        unit="step",
    ):  # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)

        # Euler step
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

        # 2nd order correction
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

        # Scale the data back
        # u_N = x_N[:, 1, :, :].unsqueeze(0)
        # u_N = ((u_N + 0.9) / 115).to(torch.float64)

        # Compute the loss
        # pde_loss, _ = get_obs_loss(u_N, u_GT, known_index_u, device=device)

        # Div 2D
        # Fx_x = torch.gradient(x_N[0, 0], dim=0)[0]
        # Fy_y = torch.gradient(x_N[0, 1], dim=1)[0]
        # div = torch.stack([Fx_x, Fy_y], dim=0)
        # pde_loss = div.sum(axis=0)
        u_N = x_N[:, 1, :, :].unsqueeze(0).to(torch.float64)
        deriv_x = (
            torch.tensor([[1, 0, -1]], dtype=torch.float64, device=device).view(
                1, 1, 1, 3
            )
            / 2
        )
        deriv_y = (
            torch.tensor([[1], [0], [-1]], dtype=torch.float64, device=device).view(
                1, 1, 3, 1
            )
            / 2
        )
        grad_x_next_x = F.conv2d(u_N, deriv_x, padding=(0, 1))
        grad_x_next_y = F.conv2d(u_N, deriv_y, padding=(1, 0))
        pde_loss = F.conv2d(grad_x_next_x, deriv_x, padding=(0, 1)) + F.conv2d(
            grad_x_next_y, deriv_y, padding=(1, 0)
        )
        L_pde = torch.norm(pde_loss, 2) / (net.img_resolution**2)
        # L_obs_u = torch.norm(observation_loss_u, 2) / 500

        # grad_x_cur_obs_u = torch.autograd.grad(
        #     outputs=L_obs_u, inputs=x_cur, retain_graph=True
        # )[0]
        grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]
        # if i <= 0.8 * num_steps:
        #     x_next = x_next  # - zeta_obs_u * grad_x_cur_obs_u
        # else:

        if i > 0.8 * num_steps:
            x_next = (
                x_next
                # - 0.1 * (zeta_obs_u * grad_x_cur_obs_u)
                - zeta_pde * grad_x_cur_pde
            )

    ############################ Save the data ############################
    # x_final = x_next
    # a_final = x_final[:, 0, :, :].unsqueeze(0)
    # u_final = x_final[:, 1, :, :].unsqueeze(0)
    # a_final = ((a_final + 1.5) / 0.2).to(torch.float64)
    # a_final[a_final > 7.5] = 12  # a is binary
    # a_final[a_final <= 7.5] = 3
    # u_final = ((u_final + 0.9) / 115).to(torch.float64)
    # relative_error_u = torch.norm(u_final - u_GT, 2) / torch.norm(u_GT, 2)
    # print(f"Relative error of u: {relative_error_u}")
    # u_final = u_final.detach().cpu().numpy()

    # scipy.io.savemat("darcy_results.mat", {"u": u_final})
    # print("Done.")
    return x_N
