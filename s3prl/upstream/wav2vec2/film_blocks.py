import torch
import torch.nn as nn
import logging

class FiLM(nn.Module):
    def __init__(self, input_size, condition_size, film_type="linear"):
        # condition_size: the size of the language id vector
        # input_size: the size of the RNN input to the FiLM layer
        super(FiLM, self).__init__()
        if film_type == "tanh":
            self.linear_scale = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
            self.linear_shift = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
        elif film_type == "linear":
            self.linear_scale = nn.Linear(condition_size, input_size)
            self.linear_shift = nn.Linear(condition_size, input_size)

    def forward(self, x, lang_condition):
        # import pdb; pdb.set_trace()
        # lang_condition = torch.permute(lang_condition, (1, 0, 2))
        if x.ndim == 3:
            gamma = self.linear_scale(lang_condition).expand_as(x)
            beta = self.linear_shift(lang_condition).expand_as(x)
            x = x * gamma + beta
        elif x.ndim == 4:
            gamma = self.linear_scale(lang_condition).unsqueeze(1).expand_as(x)
            beta = self.linear_shift(lang_condition).unsqueeze(1).expand_as(x)
            x = x * gamma + beta
        return x


class DoubleFiLM(nn.Module):
    def __init__(self, input_size, condition_size, film_type="linear"):
        # condition_size: the size of the language id vector
        # input_size: the size of the RNN input to the FiLM layer
        super(DoubleFiLM, self).__init__()
        if film_type == "tanh":
            self.linear_scale = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
            self.linear_shift = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
            self.linear_scale_2 = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
            self.linear_shift_2 = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )

        elif film_type == "linear":
            self.linear_scale = nn.Linear(condition_size, input_size)
            self.linear_shift = nn.Linear(condition_size, input_size)
            self.linear_scale_2 = nn.Linear(condition_size, input_size)
            self.linear_shift_2 = nn.Linear(condition_size, input_size)

    def forward(self, x, lang_condition):
        # import pdb; pdb.set_trace()
        # lang_condition = torch.permute(lang_condition, (1, 0, 2))
        
        if x.ndim == 3:
            # logging.info(f"lang_condition shape: {lang_condition[0].shape}")
            gamma1 = self.linear_scale(lang_condition[0]).expand_as(x)
            beta1 = self.linear_shift(lang_condition[0]).expand_as(x)
            gamma2 = self.linear_scale_2(lang_condition[1]).expand_as(x)
            beta2 = self.linear_shift_2(lang_condition[1]).expand_as(x)
            x = x * (gamma1 * gamma2) + (beta1 + beta2)
        elif x.ndim == 4:
            gamma1 = self.linear_scale(lang_condition[0]).unsqueeze(1).expand_as(x)
            beta1 = self.linear_shift(lang_condition[0]).unsqueeze(1).expand_as(x)
            gamma2 = self.linear_scale_2(lang_condition[1]).unsqueeze(1).expand_as(x)
            beta2 = self.linear_shift_2(lang_condition[1]).unsqueeze(1).expand_as(x)
            x = x * (gamma1 * gamma2) + (beta1 + beta2)
        return x



class ConditionalBatchNorm(nn.Module):
    def __init__(self, input_size, condition_size, layer_type="linear"):
        super(ConditionalBatchNorm, self).__init__()
        self.input_size = input_size
        self.bn = nn.BatchNorm1d(input_size, affine=False)
        self.linear_scale = nn.Linear(condition_size, input_size)
        self.linear_shift = nn.Linear(condition_size, input_size)
        # self.embed = nn.Linear(condition_size, input_size * 2)
        # self.embed.weight.data[:, :input_size].fill_(1)
        # self.embed.weight.data[:, input_size:].zero_()

    def forward(self, x, condition):
        # logging.info(f"input shape: {x.shape}")
        # logging.info(f"condition shape: {condition.shape}")
        # loggin.info(f"self.bn.weight shape: {self.bn.weight.shape}")
        x = torch.permute(x, (1,2,0))
        out = self.bn(x)
        out = torch.permute(out, (2,0,1))
        if x.ndim == 3:
            gamma = self.linear_scale(condition).expand_as(out)
            beta = self.linear_shift(condition).expand_as(out)
            out = out * gamma + beta
        elif x.ndim == 4:
            gamma = self.linear_scale(condition).unsqueeze(1).expand_as(out)
            beta = self.linear_shift(condition).unsqueeze(1).expand_as(out)
            out = out * gamma + beta
        return out

        # gamma, beta = self.embed(condition).chunk(2, 1)
        # gamma = gamma.unsqueeze(2).unsqueeze(3)
        # beta = beta.unsqueeze(2).unsqueeze(3)
        # return gamma * out + beta
