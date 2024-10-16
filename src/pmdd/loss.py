import torch


class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5) -> None:
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, fields) -> torch.Tensor:
        rnd_uniform = torch.rand([fields.shape[0], 1, 1, 1], device=fields.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y = fields
        n = torch.randn_like(y) * sigma
        labels = torch.zeros(fields.shape[0]).to(fields.device)
        D_yn = net(y + n, sigma, labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t) -> torch.Tensor:
        t = torch.as_tensor(t)
        return (0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1
