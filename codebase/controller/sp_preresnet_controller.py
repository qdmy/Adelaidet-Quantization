import functools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from codebase.controller.controller import bernoulli_sample

N_STAGES = 3
DEPTHS = [1]
WIDTH_MULTS = [0.5, 0.60, 0.70, 0.80, 0.90, 1.0]


class PreResNetController(nn.Module):
    def __init__(self, constraint_list=[10.0, 12.5, 15.0, 17.5, 20.0], n_superclass=20, n_stage=N_STAGES,
                 width_mults=WIDTH_MULTS, hidden_size=64, batch_size=1, device="cpu"):
        super(PreResNetController, self).__init__()
        self.n_superclass = n_superclass
        self.n_stages = n_stage
        self.n_conditions = len(constraint_list)
        # self.depths = depths
        self.width_mults = width_mults
        self.hidden_size = hidden_size

        self.register_buffer("constraint_list", torch.tensor(constraint_list))

        self.superclass_embedding = nn.Embedding(self.n_superclass, int(self.hidden_size / 2))
        self.condition_embedding = nn.Embedding(self.n_conditions, int(self.hidden_size / 2))
        # self.depth_embedding = nn.Embedding(len(self.depths), self.hidden_size)
        self.width_mult_embedding = nn.Embedding(len(self.width_mults), self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.width_linear = nn.Linear(self.hidden_size, len(self.width_mults) - 1)

        self.batch_size = batch_size
        self.device = device
        self.reset_parameters()

    def reset_parameters(self, init_range=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    @functools.lru_cache(maxsize=128)
    def _zeros(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), device=self.device, requires_grad=False)

    def sample_constraint(self):
        return random.uniform(self.constraint_list[0].item(), self.constraint_list[-1].item())

    def sample_superclass(self):
        return random.randint(0, self.n_superclass - 1)

    def linear_interpolation(self, constraints):
        result_embeddings = []
        for constraint in constraints:
            for i in range(len(self.constraint_list) - 1):
                left = self.constraint_list[i]
                right = self.constraint_list[i + 1]
                if left <= constraint <= right:
                    interpolation_w = (right - constraint) / (right - left)
                    input_idxes = self.condition_embedding.weight.new_tensor([i, i + 1], dtype=torch.long)
                    output_embedding = self.condition_embedding(input_idxes)
                    result_embedding = interpolation_w * output_embedding[0] + (
                            1 - interpolation_w) * output_embedding[1]
                    result_embeddings.append(result_embedding)
                    break
        result_embeddings = torch.stack(result_embeddings, dim=0)
        return result_embeddings

    def forward(self, constraints, superclass_id, temperature=1.0, uniform=False):
        input_constraint = self.linear_interpolation(constraints)
        superclass = self.superclass_embedding(superclass_id).expand(input_constraint.shape[0], -1)
        inputs = torch.cat((input_constraint, superclass), 1)

        hidden = self._zeros(len(constraints)), self._zeros(len(constraints))
        embed = inputs

        width_mults = []
        cum_indicators = []

        for stage in range(self.n_stages):
            # width_mults
            if uniform:
                logits = torch.zeros(len(self.width_mults))
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.width_linear(hx)
            probs = F.sigmoid(logits)
            one_indicator = probs.new_ones(probs.shape[0], 1)
            if self.training:
                # right_indicator = BernoulliSample.apply(probs)
                right_indicator = bernoulli_sample(probs, temperature)
            else:
                right_indicator = (probs > 0.5).float()
            indicator = torch.cat((one_indicator, right_indicator), dim=1)
            cum_indicator = torch.cumprod(indicator, dim=1)
            sample = (cum_indicator.sum(dim=1) - 1.0).long()

            width_mults.append(sample)
            cum_indicators.append(cum_indicator)

            embed = self.width_mult_embedding(sample.reshape(-1))
        return width_mults, cum_indicators
