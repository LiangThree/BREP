import os
import torch
import torch.nn as nn
import re
import pdb
from transformers import PreTrainedModel
target_dict = {}
total_parameter = 0

class ClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout()
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ActivationLayer(nn.Module):
    def __init__(self, hidden_size, update_layer, layer_type="all", op_position="", is_bfloat16=False, prefix=None, layer_idx=None):
        super().__init__()
        
        self.update_layer = update_layer
        self.layer_type = layer_type
        self.op_position = op_position
        self.modify_count = 0
        self.prefix = prefix
        self.layer_idx = layer_idx
        
        if(is_bfloat16):
            self.weight_type = torch.bfloat16
        else:
            self.weight_type = torch.float32
        
        if(self.layer_type=="all"):
            self.delta_vector = nn.ParameterDict({
                "activation_scaling": nn.Parameter(torch.ones(1, hidden_size)),
                "activation_bias":nn.Parameter(torch.zeros(1, hidden_size)),
            })
        elif(self.layer_type=="scaling"):
            self.delta_vector = nn.ParameterDict({
                "activation_scaling": nn.Parameter(torch.ones(1, hidden_size))
            })
        elif(self.layer_type=="bias"):
            self.delta_vector = nn.ParameterDict({
                "activation_bias":nn.Parameter(torch.zeros(1, hidden_size))
            })
        elif(self.layer_type=="ln"):
            self.delta_vector = nn.ParameterDict({
                "activation_ln": nn.LayerNorm(hidden_size),
                "activation_scaling": nn.Parameter(torch.ones(1, hidden_size)),
                "activation_bias":nn.Parameter(torch.zeros(1, hidden_size)),
            })
        
        self.weight = torch.rand(1)
        self.delta_vector.to(self.weight_type)

    def get_bias_norm(self):
        """获取当前层的activation_bias的L2范数"""
        if hasattr(self.delta_vector, 'activation_bias'):
            return torch.norm(self.delta_vector['activation_bias'], p=2)
        return torch.tensor(0.0)
    
    def forward(self, x, input_tensor=None):

        if self.op_position == "res" or self.op_position == "res_with_attn" or self.op_position == "res_with_res":
            hidden_states = self.update_layer(x, input_tensor)
        else:
            hidden_states = self.update_layer(x)
        
        if self.prefix==0:
            pass
        
        elif x.shape[1] > 1  or self.prefix==-1: # NOTE 对于promt的前向传递
            
            self.modify_count = 0
            
            # if self.layer_idx == 1:
            #     print('layer_idx:', self.layer_idx, f'{x.shape} modify')
            
            if(self.layer_type=="all"):
                self.delta_vector.to(hidden_states.device)
                hidden_states = hidden_states * self.delta_vector["activation_scaling"]
                hidden_states = hidden_states + self.delta_vector["activation_bias"]            
            elif(self.layer_type=="scaling"):
                hidden_states = hidden_states * self.delta_vector["activation_scaling"]
            elif(self.layer_type=="bias"):
                hidden_states = hidden_states + self.delta_vector["activation_bias"]
            elif(self.layer_type=="ln"):
                hidden_states = hidden_states * self.delta_vector["activation_scaling"]
                hidden_states = hidden_states + self.delta_vector["activation_bias"]
                hidden_states = self.delta_vector["activation_ln"](hidden_states)

        elif x.shape[1] == 1:
            if self.modify_count < self.prefix: # NOTE 对于前缀进行调整，超出前缀范围不再调整
            
                self.modify_count += 1

                # if self.layer_idx == 1:
                #     print('layer_idx:', self.layer_idx, f'{x.shape} modify', f'{self.modify_count} modify_count')

                if(self.layer_type=="all"):
                    self.delta_vector.to(hidden_states.device)
                    hidden_states = hidden_states * self.delta_vector["activation_scaling"]
                    hidden_states = hidden_states + self.delta_vector["activation_bias"]            
                elif(self.layer_type=="scaling"):
                    hidden_states = hidden_states * self.delta_vector["activation_scaling"]
                elif(self.layer_type=="bias"):
                    hidden_states = hidden_states + self.delta_vector["activation_bias"]
                elif(self.layer_type=="ln"):
                    hidden_states = hidden_states * self.delta_vector["activation_scaling"]
                    hidden_states = hidden_states + self.delta_vector["activation_bias"]
                    hidden_states = self.delta_vector["activation_ln"](hidden_states)
        
        if(self.op_position =="res_with_res"):
            hidden_states = hidden_states + x 
        
        return hidden_states


class ActivationModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.model_type = "t5-base"
        self.frozen_model()
        key_list = [key for key, _ in base_model.named_modules()]
        for key in key_list:
            if(self.check_update(key)):
                self.replace_layer(key)    

    def check_update(self, key):
        check_list = ["wo"]
        for name in check_list:
            if(name in key):
                return True
        return False

    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        new_module = ActivationLayer(
                        hidden_size = self.base_model.config.d_model,
                        update_layer = replaced_module)
        setattr(parent_module, replaced_name_last, new_module)


    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0 
        for name, param in self.base_model.named_parameters():
            total_parameters+=param.numel()
            if(param.requires_grad):
                trainable_parameters+=param.numel()
        
        base_model_total_parameters = total_parameters - trainable_parameters
        return trainable_parameters/base_model_total_parameters, trainable_parameters


    def frozen_model(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.base_model(input_ids, attention_mask, labels = labels)
    
    def generate(self, **args):
        return self.base_model.generate(**args)
    
    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_path)

    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module


class ActivationLLama(nn.Module):
    _no_split_modules = ["LlamaDecoderLayer"]
    def __init__(self, base_model, op_position=None, layer_type="all", exclude_layers=[], prefix=None):
        super().__init__()
        self.base_model = base_model
        self.model_type = "llama-7b"
        self.layer_type = layer_type
        self.op_position = op_position
        self.exclude_layers = exclude_layers
        self.prefix = prefix
        print('n_prefix:', prefix)

        if(exclude_layers):
            pattern_str = '|'.join(map(str, exclude_layers))
            pattern = re.compile(r'\b(?:' + pattern_str + r')\b')
        self.frozen_model()
        key_list = [key for key, _ in base_model.named_modules()]
        for key in key_list:
            if(exclude_layers):
                match = pattern.search(key)
                if(match):
                    continue
            if(self.check_update(key)):
                self.replace_layer(key)   

        print(self.print_trainable_parameters())

    def gradient_checkpointing_enable(self, **kwargs):
        return self.base_model.gradient_checkpointing_enable(**kwargs)

    def check_update(self, key):
        if(self.op_position=="ffn_down"):
            return self.match_substring(key)
        elif self.op_position == "ffn_up":
            return self.match_substring_up_proj(key)
        elif self.op_position == "attn_o":
            return self.match_substring_attn_o_proj(key)
        elif self.op_position == "attn_q":
            return self.match_substring_attn_q_proj(key)
        elif self.op_position == "attn_k":
            return self.match_substring_attn_k_proj(key)
        elif self.op_position == "attn_v":
            return self.match_substring_attn_v_proj(key)

    def generate(self, **args):
        return self.base_model.generate(**args)


    def replace_layer(self, key):
        
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        layer_idx = int(key.split(".")[2])

        if layer_idx == 1:
            print(f'---------------- replace layer ------------------')

        if "q_proj" in key or "k_proj" in key or "v_proj" or "up_proj" in key:
            hidden_size = replaced_module.out_features  # 假设这些层是线性层
        else:
            hidden_size = self.base_model.config.hidden_size

        new_module = ActivationLayer(
            hidden_size = hidden_size,
            update_layer = replaced_module,
            layer_type = self.layer_type,
            op_position = self.op_position,
            is_bfloat16=True,
            prefix=self.prefix,
            layer_idx=layer_idx,
        )
        setattr(parent_module, replaced_name_last, new_module)

    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0 
        for name, param in self.base_model.named_parameters():
            total_parameters+=param.numel()
            if(param.requires_grad):
                trainable_parameters+=param.numel()
    
        return {
            "total_para:": total_parameters,
            "trainable_para: ":trainable_parameters,
            "trainable%:" : f"{100 * trainable_parameters / total_parameters:.4f}"
            }


    def frozen_model(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False
    

    def match_substring(self, input_string):
        pattern = r'down_proj'
        return re.search(pattern, input_string) is not None
    
    def match_substring_up_proj(self, input_string):
        pattern = r'up_proj'
        return re.search(pattern, input_string) is not None
    
    def match_substring_attn_o_proj(self, input_string):
        pattern = r'o_proj'
        return re.search(pattern, input_string) is not None
    
    def match_substring_attn_q_proj(self, input_string):
        pattern = r'q_proj'
        return re.search(pattern, input_string) is not None

    def match_substring_attn_k_proj(self, input_string):
        pattern = r'k_proj'
        return re.search(pattern, input_string) is not None

    def match_substring_attn_v_proj(self, input_string):
        pattern = r'v_proj'
        return re.search(pattern, input_string) is not None

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output
    
    def get_bias_norms(self):
        """获取所有层的activation_bias的L2范数"""
        norms = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, ActivationLayer):
                norm = module.get_bias_norm()
                if norm >= 0:  # 只收集有偏置的层
                    norms.append(norm)
        return norms
    
    def get_layer_bias_norms(self):
        """获取所有层的activation_bias的L2范数"""
        norms = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, ActivationLayer) and hasattr(module.delta_vector, 'activation_bias'):
                bias = module.delta_vector['activation_bias']
                norm = torch.norm(bias, p=2).item()
                norms.append(norm)
        return norms
    
    def get_avg_bias_norm(self):
        """计算所有层activation_bias的平均L2范数"""
        norms = self.get_activation_bias_norms()
        return sum(norms) / len(norms) if norms else 0.0
    
    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                if("activation_ln" in key):
                    if("weight" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).weight.data = new_module
                    elif("bias" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).bias.data = new_module
                else:
                    self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module

    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()

        # 获取 save_path 的父目录（确保不会误把文件路径当作目录创建）
        save_dir = os.path.dirname(save_path)  
        if not os.path.exists(save_dir):  
            os.makedirs(save_dir, exist_ok=True)  # 只创建目录

        torch.save(save_dict, save_path)