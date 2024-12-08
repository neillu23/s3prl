import torch
import torch.nn as nn
import logging
import torch.nn.functional as F


class DoubleTCAC(nn.Module):
    def __init__(self, input_size, condition_size, hidden_size=128, warmup_attention_steps=None):
        super(DoubleTCAC, self).__init__()

        self.linear_scale = nn.Linear(condition_size, input_size)
        self.linear_shift = nn.Linear(condition_size, input_size)

        self.linear_scale_2 = nn.Linear(condition_size, input_size)
        self.linear_shift_2 = nn.Linear(condition_size, input_size)

        self.attention_module_time_weight = AttentionModule2(input_size, condition_size, hidden_size, init_value=1.0)
        self.attention_module_time_weight_2 = AttentionModule2(input_size, condition_size, hidden_size, init_value=1.0)
        

        self.warmup_attention_steps = warmup_attention_steps
        self._initialize_weights()

    def _initialize_weights(self):
        # Set linear_scale and linear_shift to zeros and ones
        nn.init.constant_(self.linear_scale.weight, 0)
        nn.init.constant_(self.linear_scale.bias, 1)

        nn.init.constant_(self.linear_shift.weight, 0)
        nn.init.constant_(self.linear_shift.bias, 0)

        nn.init.constant_(self.linear_scale_2.weight, 0)
        nn.init.constant_(self.linear_scale_2.bias, 1)

        nn.init.constant_(self.linear_shift_2.weight, 0)
        nn.init.constant_(self.linear_shift_2.bias, 0)

    def forward(self, x, lang_condition, step=None):
        # Calculate base gamma and beta
        # x: # [T, N, C]
        # # logging.info(f"x shape: {x.shape}")
        # # logging.info(f"lang_condition shape: {lang_condition.shape}")
        gamma = self.linear_scale(lang_condition[0]) # [1, N, C]
        beta = self.linear_shift(lang_condition[0]) # [1, N, C]
        # # logging.info(f"lang_condition shape: {lang_condition.shape}")

        # Calculate attention weights
        attention_weights_time_weight = self.attention_module_time_weight(x, lang_condition[0]) # [T, N, 1]
        
        # attention_weights_time_bias = self.attention_module_time_bias(x, lang_condition) # [T, N, 1]

        # attention_weights_channel = self.attention_module_channel(x, lang_condition)# [1, N, C]

        # Apply attention to gamma and beta
        gamma = gamma * attention_weights_time_weight # [T, N, C]
        beta = beta * attention_weights_time_weight # [T, N, C]
        
        if lang_condition[1] is not None:
            gamma_2 = self.linear_scale_2(lang_condition[1]) # [1, N, C]
            beta_2 = self.linear_shift_2(lang_condition[1]) # [1, N, C]
            attention_weights_time_weight_2 = self.attention_module_time_weight_2(x, lang_condition[1]) # [T, N, 1]
            gamma_2 = gamma_2 * attention_weights_time_weight_2 # [T, N, C]
            beta_2 = beta_2 * attention_weights_time_weight_2 # [T, N, C]
            gamma = gamma * gamma_2
            beta = beta + beta_2

        if x.ndim == 3:
            x = x * gamma + beta
        elif x.ndim == 4:
            gamma = gamma.unsqueeze(1) # For matching dimensions in case of 4D input
            beta = beta.unsqueeze(1) # For matching dimensions in case of 4D input
            x = x * gamma + beta

        # logging.info(f"x shape: {x.shape}")
        return x



class TCAC(nn.Module):
    def __init__(self, input_size, condition_size, hidden_size=128, warmup_attention_steps=None):
        super(TCAC, self).__init__()

        self.linear_scale = nn.Linear(condition_size, input_size)
        self.linear_shift = nn.Linear(condition_size, input_size)

        self.attention_module_time_weight = AttentionModule2(input_size, condition_size, hidden_size, init_value=1.0)            
        self.warmup_attention_steps = warmup_attention_steps
        self._initialize_weights()

    def _initialize_weights(self):
        # Set linear_scale and linear_shift to zeros and ones
        nn.init.constant_(self.linear_scale.weight, 0)
        nn.init.constant_(self.linear_scale.bias, 1)

        nn.init.constant_(self.linear_shift.weight, 0)
        nn.init.constant_(self.linear_shift.bias, 0)

    def forward(self, x, lang_condition, step=None):
        # Calculate base gamma and beta
        # x: # [T, N, C]
        # # logging.info(f"x shape: {x.shape}")
        # # logging.info(f"lang_condition shape: {lang_condition.shape}")
        gamma = self.linear_scale(lang_condition) # [1, N, C]
        beta = self.linear_shift(lang_condition) # [1, N, C]
        # # logging.info(f"lang_condition shape: {lang_condition.shape}")

        # Calculate attention weights
        attention_weights_time_weight = self.attention_module_time_weight(x, lang_condition) # [T, N, 1]
        # attention_weights_time_bias = self.attention_module_time_bias(x, lang_condition) # [T, N, 1]

        # attention_weights_channel = self.attention_module_channel(x, lang_condition)# [1, N, C]

        # Apply attention to gamma and beta
        gamma = gamma * attention_weights_time_weight # [T, N, C]
        beta = beta * attention_weights_time_weight # [T, N, C]


        if x.ndim == 3:
            x = x * gamma + beta
        elif x.ndim == 4:
            gamma = gamma.unsqueeze(1) # For matching dimensions in case of 4D input
            beta = beta.unsqueeze(1) # For matching dimensions in case of 4D input
            x = x * gamma + beta

        # logging.info(f"x shape: {x.shape}")
        return x


 

class AttentionModule2(nn.Module):
    def __init__(self, input_size, condition_size, hidden_size, init_value=1.0):
        super(AttentionModule2, self).__init__()
        self.attention_fc = nn.Linear(input_size + condition_size, hidden_size)
        # fix weight for all layers + lora (A*B^T)
        self.output_fc = nn.Linear(hidden_size, 1)
        # fix weight for all layers 
        # self.nonsoftmax = nonsoftmax

        self.relu = nn.PReLU() 
        

        nn.init.kaiming_uniform_(self.attention_fc.weight, nonlinearity='relu')
        nn.init.constant_(self.attention_fc.bias, 0)
        # if self.nonsoftmax:
        nn.init.constant_(self.output_fc.weight, 0)
        nn.init.constant_(self.output_fc.bias, init_value)
        # else:
        #     nn.init.kaiming_uniform_(self.output_fc.weight)
        #     nn.init.constant_(self.output_fc.bias, 0)
        #     #xavier_uniform_


    def forward(self, x, lang_condition, step=None):
        # x: [T, N, C], lang_condition: [1, N, C]
        T, N, C = x.size()
        # logging.info(f"x shape: {x.shape}")
        lang_condition = lang_condition.repeat(T, 1, 1) # [T, N, C]
        combined_input = torch.cat((x, lang_condition), dim=-1) # [T, N, 2*C]

        attention_scores = self.attention_fc(combined_input)
        attention_scores = self.relu(attention_scores) # [T, N,  hidden_size]
        attention_scores = self.output_fc(attention_scores).squeeze(-1) # [T, N]

        # if not self.nonsoftmax:
        #     attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1) # [T, N, 1]
        # else:
        attention_weights = attention_scores.unsqueeze(-1)
        return attention_weights


class CC(nn.Module):
    def __init__(self, input_size, condition_size, act_type="linear"):
        # condition_size: the size of the language id vector
        # input_size: the size of the RNN input to the CC layer
        super(CC, self).__init__()
        if act_type == "tanh":
            self.linear_scale = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
            self.linear_shift = nn.Sequential(
                nn.Linear(condition_size, input_size),
                nn.Tanh()
            )
        elif act_type == "linear":
            self.linear_scale = nn.Linear(condition_size, input_size)
            self.linear_shift = nn.Linear(condition_size, input_size)
        self._initialize_weights()

    def _initialize_weights(self):
        # Set linear_scale and linear_shift to zeros and ones
        nn.init.constant_(self.linear_scale.weight, 0)
        nn.init.constant_(self.linear_scale.bias, 1)

        nn.init.constant_(self.linear_shift.weight, 0)
        nn.init.constant_(self.linear_shift.bias, 0)

    def forward(self, x, lang_condition, step=None):
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


class DoubleCC(nn.Module):
    def __init__(self, input_size, condition_size, act_type="linear"):
        # condition_size: the size of the language id vector
        # input_size: the size of the RNN input to the CC layer
        super(DoubleCC, self).__init__()
        if act_type == "tanh":
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

        elif act_type == "linear":
            self.linear_scale = nn.Linear(condition_size, input_size)
            self.linear_shift = nn.Linear(condition_size, input_size)
            self.linear_scale_2 = nn.Linear(condition_size, input_size)
            self.linear_shift_2 = nn.Linear(condition_size, input_size)
        self._initialize_weights()

    def _initialize_weights(self):
        # Set linear_scale and linear_shift to zeros and ones
        nn.init.constant_(self.linear_scale.weight, 0)
        nn.init.constant_(self.linear_scale.bias, 1)

        nn.init.constant_(self.linear_shift.weight, 0)
        nn.init.constant_(self.linear_shift.bias, 0)

        nn.init.constant_(self.linear_scale_2.weight, 0)
        nn.init.constant_(self.linear_scale_2.bias, 1)

        nn.init.constant_(self.linear_shift_2.weight, 0)
        nn.init.constant_(self.linear_shift_2.bias, 0)

    def forward(self, x, lang_condition, step=None):
        # import pdb; pdb.set_trace()
        # lang_condition = torch.permute(lang_condition, (1, 0, 2))
        
        if x.ndim == 3:
            # logging.info(f"lang_condition shape: {lang_condition[0].shape}")
            gamma1 = self.linear_scale(lang_condition[0]).expand_as(x)
            beta1 = self.linear_shift(lang_condition[0]).expand_as(x)
            if lang_condition[1] is not None:
                gamma2 = self.linear_scale_2(lang_condition[1]).expand_as(x)
                beta2 = self.linear_shift_2(lang_condition[1]).expand_as(x)
                x = x * (gamma1 * gamma2) + (beta1 + beta2)
            else:
                x = x * gamma1 + beta1
        elif x.ndim == 4:
            gamma1 = self.linear_scale(lang_condition[0]).unsqueeze(1).expand_as(x)
            beta1 = self.linear_shift(lang_condition[0]).unsqueeze(1).expand_as(x)
            if lang_condition[1] is not None:
                gamma2 = self.linear_scale_2(lang_condition[1]).unsqueeze(1).expand_as(x)
                beta2 = self.linear_shift_2(lang_condition[1]).unsqueeze(1).expand_as(x)
                x = x * (gamma1 * gamma2) + (beta1 + beta2)
            else:
                x = x * gamma1 + beta1
        return x

