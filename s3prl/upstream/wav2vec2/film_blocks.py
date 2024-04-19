import torch
import torch.nn as nn
import logging
import torch.nn.functional as F


class AttFiLM2(nn.Module):
    def __init__(self, input_size, condition_size, hidden_size=128, warmup_attention_steps=None, nonsoftmax=False, film_type="linear", att_type="att2", relu2=False):
        super(AttFiLM2, self).__init__()

        self.linear_scale = nn.Linear(condition_size, input_size)
        self.linear_shift = nn.Linear(condition_size, input_size)


        if att_type == "att2":
            self.attention_module_time_weight = AttentionModule2(input_size, condition_size, hidden_size, nonsoftmax=nonsoftmax, init_value=1.0)
            # self.attention_module_channel_weight = AttentionModule2(input_size, condition_size, hidden_size, nonsoftmax=nonsoftmax, init_value=1.0)
            self.attention_module_time_bias = AttentionModule2(input_size, condition_size, hidden_size, nonsoftmax=nonsoftmax, init_value=1.0)
            # self.attention_module_channel_bias = AttentionModule2(input_size, condition_size, hidden_size, nonsoftmax=nonsoftmax, init_value=1.0)
        if att_type == "gatedatt":
            self.attention_module_time_weight = GatedAttentionModule(input_size, condition_size, hidden_size, init_value=1.0)
            # self.attention_module_channel_weight = GatedAttentionModule(input_size, condition_size, hidden_size, nonsoftmax=nonsoftmax, init_value=1.0)
            self.attention_module_time_bias = GatedAttentionModule(input_size, condition_size, hidden_size, init_value=1.0)
            # self.attention_module_channel_bias = GatedAttentionModule(input_size, condition_size, hidden_size, nonsoftmax=nonsoftmax, init_value=1.0)
            

        self.warmup_attention_steps = warmup_attention_steps

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
        attention_weights_time_bias = self.attention_module_time_bias(x, lang_condition) # [T, N, 1]

        # attention_weights_channel = self.attention_module_channel(x, lang_condition)# [1, N, C]

        # Apply attention to gamma and beta
        gamma = gamma * attention_weights_time_weight # [T, N, C]
        beta = beta * attention_weights_time_bias # [T, N, C]


        if x.ndim == 3:
            x = x * gamma + beta
        elif x.ndim == 4:
            gamma = gamma.unsqueeze(1) # For matching dimensions in case of 4D input
            beta = beta.unsqueeze(1) # For matching dimensions in case of 4D input
            x = x * gamma + beta

        # logging.info(f"x shape: {x.shape}")
        return x




class GatedAttentionModule(nn.Module):
    def __init__(self, input_size, condition_size, hidden_size, init_value=1.0):
        super(GatedAttentionModule, self).__init__()
        self.attention_fc = nn.Linear(input_size + condition_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, 1)

        self.filter_gate = nn.Linear(hidden_size, hidden_size)
        self.gating_gate = nn.Linear(hidden_size, hidden_size)
        

        nn.init.kaiming_uniform_(self.attention_fc.weight, nonlinearity='relu')
        nn.init.constant_(self.attention_fc.bias, 0)
        # init output_fc as 1 
        nn.init.constant_(self.output_fc.weight, 0)
        nn.init.constant_(self.output_fc.bias, init_value)


    def forward(self, x, lang_condition, step=None):
        # x: [T, N, C], lang_condition: [1, N, C]
        T, N, C = x.size()
        lang_condition = lang_condition.repeat(T, 1, 1) # [T, N, C]
        combined_input = torch.cat((x, lang_condition), dim=-1) # [T, N, 2*C]

        attention_scores = self.attention_fc(combined_input) # [T, N,  hidden_size]
        gated_hidden = torch.tanh(self.filter_gate(attention_scores)) * torch.sigmoid(self.gating_gate(attention_scores)) # [T, N,  hidden_size]
        attention_scores = self.output_fc(gated_hidden).squeeze(-1) # [T, N]

        attention_weights = attention_scores.unsqueeze(-1)
        return attention_weights





class AttentionModule2(nn.Module):
    def __init__(self, input_size, condition_size, hidden_size, nonsoftmax=False, init_value=1.0):
        super(AttentionModule2, self).__init__()
        self.attention_fc = nn.Linear(input_size + condition_size, hidden_size)
        # fix weight for all layers + lora (A*B^T)
        self.output_fc = nn.Linear(hidden_size, 1)
        # fix weight for all layers 
        self.nonsoftmax = nonsoftmax

        self.relu = nn.PReLU() 
        

        nn.init.kaiming_uniform_(self.attention_fc.weight, nonlinearity='relu')
        nn.init.constant_(self.attention_fc.bias, 0)
        if self.nonsoftmax:
            nn.init.constant_(self.output_fc.weight, 0)
            nn.init.constant_(self.output_fc.bias, init_value)
        else:
            nn.init.kaiming_uniform_(self.output_fc.weight)
            nn.init.constant_(self.output_fc.bias, 0)
            #xavier_uniform_


    def forward(self, x, lang_condition, step=None):
        # x: [T, N, C], lang_condition: [1, N, C]
        T, N, C = x.size()
        lang_condition = lang_condition.repeat(T, 1, 1) # [T, N, C]
        combined_input = torch.cat((x, lang_condition), dim=-1) # [T, N, 2*C]

        attention_scores = self.attention_fc(combined_input)
        attention_scores = self.relu(attention_scores) # [T, N,  hidden_size]
        attention_scores = self.output_fc(attention_scores).squeeze(-1) # [T, N]

        if not self.nonsoftmax:
            attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1) # [T, N, 1]
        else:
            attention_weights = attention_scores.unsqueeze(-1)
        return attention_weights


class AttentionModule(nn.Module):
    def __init__(self, input_size, condition_size, hidden_size, nonsoftmax=False, dropout_rate=0.3, relu2=False):
        super(AttentionModule, self).__init__()
        self.attention_fc = nn.Linear(input_size + condition_size, hidden_size)
        # fix weight for all layers + lora (A*B^T)
        self.output_fc = nn.Linear(hidden_size, 1)
        # fix weight for all layers 
        self.nonsoftmax = nonsoftmax

        self.layer_norm = nn.LayerNorm(hidden_size)  
        self.dropout = nn.Dropout(dropout_rate)  
        self.relu = nn.PReLU() 
        if relu2:
            self.relu2 = nn.PReLU() 

        nn.init.kaiming_uniform_(self.attention_fc.weight, nonlinearity='relu')
        nn.init.constant_(self.attention_fc.bias, 0)
        if self.nonsoftmax:
            nn.init.constant_(self.output_fc.weight, 0)
            nn.init.constant_(self.output_fc.bias, 1)
        else:
            nn.init.kaiming_uniform_(self.output_fc.weight)
            #xavier_uniform_
            nn.init.constant_(self.output_fc.bias, 0)


    def forward(self, x, lang_condition, step=None):
        # x: [T, N, C], lang_condition: [1, N, C]
        T, N, C = x.size()
        lang_condition = lang_condition.repeat(T, 1, 1) # [T, N, C]
        combined_input = torch.cat((x, lang_condition), dim=-1) # [T, N, 2*C]

        attention_scores = self.attention_fc(combined_input)
        attention_scores = self.layer_norm(attention_scores)
        attention_scores = self.relu(attention_scores) # [T, N,  hidden_size]
        attention_scores = self.dropout(attention_scores) 

        attention_scores = self.output_fc(attention_scores).squeeze(-1) # [T, N]
        if hasattr(self, 'relu2'):
            attention_scores = self.relu2(attention_scores)

        if not self.nonsoftmax:
            attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1) # [T, N, 1]
        else:
            attention_weights = attention_scores.unsqueeze(-1)
        return attention_weights

class AttFiLM(nn.Module):
    def __init__(self, input_size, condition_size, hidden_size=128, warmup_attention_steps=None, nonsoftmax=False, film_type="linear", att_type="att1", relu2=False):
        super(AttFiLM, self).__init__()
        if film_type == "linear":
            self.linear_scale = nn.Linear(condition_size, input_size)
            self.linear_shift = nn.Linear(condition_size, input_size)
        if att_type == "att1":
            self.attention_module = AttentionModule(input_size, condition_size, hidden_size, nonsoftmax=nonsoftmax,relu2=relu2)
        if att_type == "att2":
            self.attention_module = AttentionModule2(input_size, condition_size, hidden_size, nonsoftmax=nonsoftmax)
        if att_type == "gatedatt":
            self.attention_module = GatedAttentionModule(input_size, condition_size, hidden_size)

        self.warmup_attention_steps = warmup_attention_steps

    def forward(self, x, lang_condition, step=None):
        # Calculate base gamma and beta
        # x: # [T, N, C]
        # logging.info(f"x shape: {x.shape}")
        # logging.info(f"lang_condition shape: {lang_condition.shape}")
        gamma = self.linear_scale(lang_condition) # [1, N, C]
        beta = self.linear_shift(lang_condition) # [1, N, C]
        # logging.info(f"lang_condition shape: {lang_condition.shape}")

        # Calculate attention weights
        attention_weights = self.attention_module(x, lang_condition) # [T, N, 1]

        # Apply attention to gamma and beta
        if self.warmup_attention_steps is None or step >= self.warmup_attention_steps:
            gamma = gamma * attention_weights # [T, N, C]
            beta = beta * attention_weights # [T, N, C]
        else:
            lambda_value = step / self.warmup_attention_steps
            adjusted_attention_weights = 1 + lambda_value * (attention_weights - 1)
            gamma = gamma * adjusted_attention_weights
            beta = beta * adjusted_attention_weights


        if x.ndim == 3:
            x = x * gamma + beta
        elif x.ndim == 4:
            gamma = gamma.unsqueeze(1) # For matching dimensions in case of 4D input
            beta = beta.unsqueeze(1) # For matching dimensions in case of 4D input
            x = x * gamma + beta

        # logging.info(f"x shape: {x.shape}")
        return x

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



class ConditionalBatchNorm(nn.Module):
    def __init__(self, input_size, condition_size, film_type="linear"):
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
