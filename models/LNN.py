import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


class CfC(nn.Module):
    def __init__(
        self,
        input_size: Union[int, "Wiring"],
        units,
        proj_size: Optional[int] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[int] = None,
    ):
        super(CfC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.proj_size = proj_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        # 关键修改：正确判断是否为 Wiring 类型
        if hasattr(units, 'units') and hasattr(units, 'output_dim'):
            # 这是 Wiring 对象（如 AutoNCP）
            self.wired_mode = True
            if backbone_units is not None:
                raise ValueError(f"Cannot use backbone_units in wired mode")
            if backbone_layers is not None:
                raise ValueError(f"Cannot use backbone_layers in wired mode")
            if backbone_dropout is not None:
                raise ValueError(f"Cannot use backbone_dropout in wired mode")
            self.wiring = units
            self.state_size = self.wiring.units
            self.output_size = self.wiring.output_dim
            self.rnn_cell = WiredCfCCell(
                input_size,
                self.wiring_or_units,
                mode,
            )
        else:
            # 普通模式，units 是整数
            self.wired_mode = False
            backbone_units = 128 if backbone_units is None else backbone_units
            backbone_layers = 1 if backbone_layers is None else backbone_layers
            backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout
            self.state_size = units
            self.output_size = self.state_size
            self.rnn_cell = CfCCell(
                input_size,
                self.wiring_or_units,
                mode,
                activation,
                backbone_units,
                backbone_layers,
                backbone_dropout,
            )
        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = LSTMCell(input_size, self.state_size)

        if proj_size is None:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.output_size, self.proj_size)

    def forward(self, input, hx=None, timespans=None):
        """
        :param input: 输入张量，无批次模式下形状为 (L,C)，batch_first=True 时为 (B,L,C)，batch_first=False 时为 (L,B,C)
        :param hx: RNN 的初始隐藏状态，mixed_memory=False 时形状为 (B,H)，mixed_memory=True 时为元组 ((B,H),(B,H))。如果为 None，隐藏状态将初始化为全零。
        :param timespans: 时间跨度
        :return: 返回一对 (output, hx)，其中 output 为输出序列，hx 为 RNN 的最终隐藏状态
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = (
                torch.zeros((batch_size, self.state_size), device=device)
                if self.use_mixed
                else None
            )
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError(
                    "Running a CfC with mixed_memory=True, requires a tuple (h0,c0) to be passed as state (got torch.Tensor instead)"
                )
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if is_batched:
                if h_state.dim() != 2:
                    msg = (
                        "For batched 2-D input, hx and cx should "
                        f"also be 2-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
            else:
                # 无批次模式
                if h_state.dim() != 1:
                    msg = (
                        "For unbatched 1-D input, hx and cx should "
                        f"also be 1-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0) if c_state is not None else None

        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if self.return_sequences:
                output_sequence.append(self.fc(h_out))

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = self.fc(h_out)
        hx = (h_state, c_state) if self.use_mixed else h_state

        if not is_batched:
            # 无批次模式
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]

        return readout, hx
    
    
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        output_state, cell_state = states
        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate

        return output_state, new_cell
    
class WiredCfCCell(nn.Module):
    def __init__(
        self,
        input_size,
        wiring,
        mode="default",
    ):
        super(WiredCfCCell, self).__init__()

        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        self._wiring = wiring

        self._layers = []
        in_features = wiring.input_dim
        for l in range(wiring.num_layers):
            hidden_units = self._wiring.get_neurons_of_layer(l)
            if l == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
            input_sparsity = np.concatenate(
                [
                    input_sparsity,
                    np.ones((len(hidden_units), len(hidden_units))),
                ],
                axis=0,
            )

            # 技巧：nn.Module 在 set_attribute 中注册子参数
            rnn_cell = CfCCell(
                in_features,
                len(hidden_units),
                mode,
                backbone_activation="lecun_tanh",
                backbone_units=0,
                backbone_layers=0,
                backbone_dropout=0.0,
                sparsity_mask=input_sparsity,
            )
            self.register_module(f"layer_{l}", rnn_cell)
            self._layers.append(rnn_cell)
            in_features = len(hidden_units)

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def layer_sizes(self):
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self):
        return self._wiring.num_layers

    def forward(self, input, hx, timespans):
        h_state = torch.split(hx, self.layer_sizes, dim=1)

        new_h_state = []
        inputs = input
        for i in range(self.num_layers):
            h, _ = self._layers[i].forward(inputs, h_state[i], timespans)
            inputs = h
            new_h_state.append(h)

        new_h_state = torch.cat(new_h_state, dim=1)
        return h, new_h_state


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfCCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        mode="default",
        backbone_activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.0,
        sparsity_mask=None,
    ):
        """闭式连续时间 (Closed-form Continuous-time) 单元。
        参考论文: https://arxiv.org/abs/2106.13898
        .. Note::
            这是一个处理单个时间步的 RNNCell。如需处理序列的完整 RNN，请参阅 `ncps.torch.CfC`。
        :param input_size: 输入特征维度
        :param hidden_size: 隐藏层大小
        :param mode: 模式 ("default", "pure", "no_gate")
        :param backbone_activation: 骨干网络激活函数
        :param backbone_units: 骨干网络单元数
        :param backbone_layers: 骨干网络层数
        :param backbone_dropout: 骨干网络 Dropout 率
        :param sparsity_mask: 稀疏性掩码
        """

        super(CfCCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                f"Unknown mode '{mode}', valid options are {str(allowed_modes)}"
            )
        self.sparsity_mask = (
            None
            if sparsity_mask is None
            else torch.nn.Parameter(
                data=torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32)),
                requires_grad=False,
            )
        )

        self.mode = mode

        if backbone_activation == "silu":
            backbone_activation = nn.SiLU
        elif backbone_activation == "relu":
            backbone_activation = nn.ReLU
        elif backbone_activation == "tanh":
            backbone_activation = nn.Tanh
        elif backbone_activation == "gelu":
            backbone_activation = nn.GELU
        elif backbone_activation == "lecun_tanh":
            backbone_activation = LeCun
        else:
            raise ValueError(f"Unknown activation {backbone_activation}")

        self.backbone = None
        self.backbone_layers = backbone_layers
        if backbone_layers > 0:
            layer_list = [
                nn.Linear(input_size + hidden_size, backbone_units),
                backbone_activation(),
            ]
            for i in range(1, backbone_layers):
                layer_list.append(nn.Linear(backbone_units, backbone_units))
                layer_list.append(backbone_activation())
                if backbone_dropout > 0.0:
                    layer_list.append(torch.nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        cat_shape = int(
            self.hidden_size + input_size if backbone_layers == 0 else backbone_units
        )

        self.ff1 = nn.Linear(cat_shape, hidden_size)
        if self.mode == "pure":
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_size), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_size), requires_grad=True
            )
        else:
            self.ff2 = nn.Linear(cat_shape, hidden_size)
            self.time_a = nn.Linear(cat_shape, hidden_size)
            self.time_b = nn.Linear(cat_shape, hidden_size)
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input, hx, ts):
        x = torch.cat([input, hx], 1)
        if self.backbone_layers > 0:
            x = self.backbone(x)
        if self.sparsity_mask is not None:
            ff1 = F.linear(x, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        else:
            ff1 = self.ff1(x)
        if self.mode == "pure":
            # 纯模式解
            new_hidden = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # CfC 模式
            if self.sparsity_mask is not None:
                ff2 = F.linear(x, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
            else:
                ff2 = self.ff2(x)
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden, new_hidden


class Wiring:
    def __init__(self, units):
        self.units = units
        self.adjacency_matrix = np.zeros([units, units], dtype=np.int32)
        self.sensory_adjacency_matrix = None
        self.input_dim = None
        self.output_dim = None

    @property
    def num_layers(self):
        return 1

    def get_neurons_of_layer(self, layer_id):
        return list(range(self.units))

    def is_built(self):
        return self.input_dim is not None

    def build(self, input_dim):
        if not self.input_dim is None and self.input_dim != input_dim:
            raise ValueError(
                "Conflicting input dimensions provided. set_input_dim() was called with {} but actual input has dimension {}".format(
                    self.input_dim, input_dim
                )
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = np.zeros(
            [input_dim, self.units], dtype=np.int32
        )

    def set_output_dim(self, output_dim):
        self.output_dim = output_dim

    # 可被子类重写
    def get_type_of_neuron(self, neuron_id):
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src, dest, polarity):
        if src < 0 or src >= self.units:
            raise ValueError(
                "Cannot add synapse originating in {} if cell has only {} units".format(
                    src, self.units
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self.units
                )
            )
        if not polarity in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self.adjacency_matrix[src, dest] = polarity

    def add_sensory_synapse(self, src, dest, polarity):
        if self.input_dim is None:
            raise ValueError(
                "Cannot add sensory synapses before build() has been called!"
            )
        if src < 0 or src >= self.input_dim:
            raise ValueError(
                "Cannot add sensory synapse originating in {} if input has only {} features".format(
                    src, self.input_dim
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self.units
                )
            )
        if not polarity in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self.sensory_adjacency_matrix[src, dest] = polarity


class NCP(Wiring):
    def __init__(
        self,
        inter_neurons,
        command_neurons,
        motor_neurons,
        sensory_fanout,
        inter_fanout,
        recurrent_command_synapses,
        motor_fanin,
        seed=22222,
    ):
        """
        创建神经回路策略 (Neural Circuit Policies) 接线。
        神经元总数 (= RNN 状态大小) 等于中间神经元、命令神经元和运动神经元的总和。
        如需更简便的方式生成 NCP 接线，请参阅 ``AutoNCP`` 接线类。

        :param inter_neurons: 中间神经元数量 (第 2 层)
        :param command_neurons: 命令神经元数量 (第 3 层)
        :param motor_neurons: 运动神经元数量 (第 4 层 = 输出数量)
        :param sensory_fanout: 从感知神经元到中间神经元的平均出边突触数
        :param inter_fanout: 从中间神经元到命令神经元的平均出边突触数
        :param recurrent_command_synapses: 命令神经元层的平均循环连接数
        :param motor_fanin: 运动神经元从命令神经元接收的平均入边突触数
        :param seed: 用于生成接线的随机种子
        """

        super(NCP, self).__init__(inter_neurons + command_neurons + motor_neurons)
        self.set_output_dim(motor_neurons)
        self._rng = np.random.RandomState(seed)
        self._num_inter_neurons = inter_neurons
        self._num_command_neurons = command_neurons
        self._num_motor_neurons = motor_neurons
        self._sensory_fanout = sensory_fanout
        self._inter_fanout = inter_fanout
        self._recurrent_command_synapses = recurrent_command_synapses
        self._motor_fanin = motor_fanin

        # 神经元 ID 排列: [0..运动神经元 ... 命令神经元 ... 中间神经元]
        self._motor_neurons = list(range(0, self._num_motor_neurons))
        self._command_neurons = list(
            range(
                self._num_motor_neurons,
                self._num_motor_neurons + self._num_command_neurons,
            )
        )
        self._inter_neurons = list(
            range(
                self._num_motor_neurons + self._num_command_neurons,
                self._num_motor_neurons
                + self._num_command_neurons
                + self._num_inter_neurons,
            )
        )

        if self._motor_fanin > self._num_command_neurons:
            raise ValueError(
                "Error: Motor fanin parameter is {} but there are only {} command neurons".format(
                    self._motor_fanin, self._num_command_neurons
                )
            )
        if self._sensory_fanout > self._num_inter_neurons:
            raise ValueError(
                "Error: Sensory fanout parameter is {} but there are only {} inter neurons".format(
                    self._sensory_fanout, self._num_inter_neurons
                )
            )
        if self._inter_fanout > self._num_command_neurons:
            raise ValueError(
                "Error:: Inter fanout parameter is {} but there are only {} command neurons".format(
                    self._inter_fanout, self._num_command_neurons
                )
            )

    @property
    def num_layers(self):
        return 3

    def get_neurons_of_layer(self, layer_id):
        if layer_id == 0:
            return self._inter_neurons
        elif layer_id == 1:
            return self._command_neurons
        elif layer_id == 2:
            return self._motor_neurons
        raise ValueError("Unknown layer {}".format(layer_id))

    def get_type_of_neuron(self, neuron_id):
        if neuron_id < self._num_motor_neurons:
            return "motor"
        if neuron_id < self._num_motor_neurons + self._num_command_neurons:
            return "command"
        return "inter"

    def _build_sensory_to_inter_layer(self):
        unreachable_inter_neurons = [l for l in self._inter_neurons]
        # 随机将每个感知神经元连接到恰好 _sensory_fanout 个中间神经元
        for src in self._sensory_neurons:
            for dest in self._rng.choice(
                self._inter_neurons, size=self._sensory_fanout, replace=False
            ):
                if dest in unreachable_inter_neurons:
                    unreachable_inter_neurons.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

        # 如果某些中间神经元未被连接，现在将其连接
        mean_inter_neuron_fanin = int(
            self._num_sensory_neurons * self._sensory_fanout / self._num_inter_neurons
        )
        # 将"被遗忘"的中间神经元连接到至少 1 个、至多所有感知神经元
        mean_inter_neuron_fanin = np.clip(
            mean_inter_neuron_fanin, 1, self._num_sensory_neurons
        )
        for dest in unreachable_inter_neurons:
            for src in self._rng.choice(
                self._sensory_neurons, size=mean_inter_neuron_fanin, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

    def _build_inter_to_command_layer(self):
        # 随机将中间神经元连接到命令神经元
        unreachable_command_neurons = [l for l in self._command_neurons]
        for src in self._inter_neurons:
            for dest in self._rng.choice(
                self._command_neurons, size=self._inter_fanout, replace=False
            ):
                if dest in unreachable_command_neurons:
                    unreachable_command_neurons.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        # 如果某些命令神经元未被连接，现在将其连接
        mean_command_neurons_fanin = int(
            self._num_inter_neurons * self._inter_fanout / self._num_command_neurons
        )
        # 将"被遗忘"的命令神经元连接到至少 1 个、至多所有中间神经元
        mean_command_neurons_fanin = np.clip(
            mean_command_neurons_fanin, 1, self._num_command_neurons
        )
        for dest in unreachable_command_neurons:
            for src in self._rng.choice(
                self._inter_neurons, size=mean_command_neurons_fanin, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def _build_recurrent_command_layer(self):
        # 在命令神经元中添加循环连接
        for i in range(self._recurrent_command_synapses):
            src = self._rng.choice(self._command_neurons)
            dest = self._rng.choice(self._command_neurons)
            polarity = self._rng.choice([-1, 1])
            self.add_synapse(src, dest, polarity)

    def _build_command__to_motor_layer(self):
        # 随机将命令神经元连接到运动神经元
        unreachable_command_neurons = [l for l in self._command_neurons]
        for dest in self._motor_neurons:
            for src in self._rng.choice(
                self._command_neurons, size=self._motor_fanin, replace=False
            ):
                if src in unreachable_command_neurons:
                    unreachable_command_neurons.remove(src)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        # 如果某些命令神经元未被连接，现在将其连接
        mean_command_fanout = int(
            self._num_motor_neurons * self._motor_fanin / self._num_command_neurons
        )
        # 将"被遗忘"的命令神经元连接到至少 1 个、至多所有运动神经元
        mean_command_fanout = np.clip(mean_command_fanout, 1, self._num_motor_neurons)
        for src in unreachable_command_neurons:
            for dest in self._rng.choice(
                self._motor_neurons, size=mean_command_fanout, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        self._num_sensory_neurons = self.input_dim
        self._sensory_neurons = list(range(0, self._num_sensory_neurons))

        self._build_sensory_to_inter_layer()
        self._build_inter_to_command_layer()
        self._build_recurrent_command_layer()
        self._build_command__to_motor_layer()


class AutoNCP(NCP):
    def __init__(
        self,
        units,
        output_size,
        sparsity_level=0.5,
        seed=7,
    ):
        """只需指定神经元总数和输出数量即可实例化 NCP 接线

        :param units: 神经元总数
        :param output_size: 运动神经元数量 (= 输出大小)。此值必须小于 units-2 (通常建议设置为神经元总数的 0.3 倍)
        :param sparsity_level: 稀疏度超参数，范围在 0.0 (非常密集) 到 0.9 (非常稀疏) 之间
        :param seed: 用于生成接线的随机种子
        """
        self._output_size = output_size
        self._sparsity_level = sparsity_level
        self._seed = seed
        if output_size >= units - 2:
            raise ValueError(
                f"Output size must be less than the number of units-2 (given {units} units, {output_size} output size)"
            )
        if sparsity_level < 0.1 or sparsity_level > 1.0:
            raise ValueError(
                f"Sparsity level must be between 0.0 and 0.9 (given {sparsity_level})"
            )
        density_level = 1.0 - sparsity_level
        inter_and_command_neurons = units - output_size
        command_neurons = max(int(0.4 * inter_and_command_neurons), 1)
        inter_neurons = inter_and_command_neurons - command_neurons

        sensory_fanout = max(int(inter_neurons * density_level), 1)
        inter_fanout = max(int(command_neurons * density_level), 1)
        recurrent_command_synapses = max(int(command_neurons * density_level * 2), 1)
        motor_fanin = max(int(command_neurons * density_level), 1)
        super(AutoNCP, self).__init__(
            inter_neurons,
            command_neurons,
            output_size,
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin,
            seed=seed,
        )


# 注意力统计池化 (Attentive Statistics Pooling, ASP)
class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        self.statistical_attention = nn.Sequential(
            nn.Conv1d(input_dim, bottleneck_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )

    # Input: (B, C, T) / Output: (B, 2*C)
    def forward(self, x):
        alpha = self.statistical_attention(x)
        mean = torch.sum(alpha * x, dim=2)
        residuals = x - mean.unsqueeze(2)
        standard_deviation = torch.sum(alpha * (residuals ** 2), dim=2)
        std = torch.sqrt(torch.clamp(standard_deviation, min=1e-6))
        return torch.cat([mean, std], dim=1)


# 双分辨率池化 (Dual-Resolution Pooling, DRASP)
class DRASP(nn.Module):
    def __init__(self, input_dim, segment_len=5):
        super().__init__()
        self.input_dim = input_dim
        self.segment_len = segment_len
        # Global Branch: Focus on overall acoustic features
        self.global_asp = AttentiveStatisticsPooling(input_dim)
        # Local Branch: Focus on transient events
        self.local_asp = AttentiveStatisticsPooling(input_dim, bottleneck_dim=16)

    # Input: (B, C, T) / Output: (B, 4*C)
    def forward(self, x):
        B, C, T = x.shape
        # Global Stats (B, 2C)
        global_stats = self.global_asp(x)
        # Local Stats (B, 2C)
        pad_len = (self.segment_len - (T % self.segment_len)) % self.segment_len
        x_pad = F.pad(x, (0, pad_len)) if pad_len > 0 else x
        # Segmentation: (B, C, T_pad) -> (B, C, N_seg, Seg_Len) -> (B * N_seg, C, Seg_Len)
        x_segmented = x_pad.view(B, C, -1, self.segment_len).permute(0, 2, 1, 3).reshape(-1, C, self.segment_len)
        # Apply ASP to each segment -> (B * N_seg, 2C)
        local_stats_all = self.local_asp(x_segmented)
        # Aggregation: (B, N_seg, 2C)
        local_stats_view = local_stats_all.view(B, -1, 2 * C)
        # Max Pooling: Extract the most prominent segment features (highlight moments)
        local_stats_max, _ = torch.max(local_stats_view, dim=1)
        return torch.cat([global_stats, local_stats_max], dim=1)


# 双向并行交叉切片CfC (Bidirectional Parallel Cross-Slice CfC, BPCSCfC)
class BiParallelCrossSliceCfC(nn.Module):
    def __init__(self, input_size, wiring_units, output_size, seq_len, window_size):
        super(BiParallelCrossSliceCfC, self).__init__()
        self.seq_len = seq_len
        self.window_size = window_size
        self.num_windows = seq_len // window_size
        # AutoNCP: automatically generate sparse connections
        self.fwd_local_wiring = AutoNCP(wiring_units, output_size)
        self.fwd_local_cfc = CfC(input_size, self.fwd_local_wiring, return_sequences=True, batch_first=True)
        self.fwd_global_wiring = AutoNCP(wiring_units, output_size)
        self.fwd_global_cfc = CfC(input_size, self.fwd_global_wiring, return_sequences=True, batch_first=True)
        self.bwd_local_wiring = AutoNCP(wiring_units, output_size)
        self.bwd_local_cfc = CfC(input_size, self.bwd_local_wiring, return_sequences=True, batch_first=True)
        self.bwd_global_wiring = AutoNCP(wiring_units, output_size)
        self.bwd_global_cfc = CfC(input_size, self.bwd_global_wiring, return_sequences=True, batch_first=True)
        # Temporal fusion: [forward local, forward global, backward local, backward global]
        self.fusion = nn.Linear(output_size * 4, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        self.act = nn.GELU()

    def _process_stream(self, x, local_cfc, global_cfc):
        B, L, C = x.shape
        W, N = self.window_size, self.num_windows
        # Local Branch
        x_local = x.view(B, N, W, C).reshape(B * N, W, C)
        out_local, _ = local_cfc(x_local)
        out_local = out_local.reshape(B, N, W, -1).view(B, L, -1)
        # Global Branch
        x_global = x.view(B, N, W, C).permute(0, 2, 1, 3).reshape(B * W, N, C)
        out_global, _ = global_cfc(x_global)
        out_global = out_global.reshape(B, W, N, -1).permute(0, 2, 1, 3).reshape(B, L, -1)
        return out_local, out_global

    def forward(self, x):
        x_flipped = torch.flip(x, [1])
        fwd_local, fwd_global = self._process_stream(x, self.fwd_local_cfc, self.fwd_global_cfc)
        bwd_local_flipped, bwd_global_flipped = self._process_stream(x_flipped, self.bwd_local_cfc, self.bwd_global_cfc)
        bwd_local = torch.flip(bwd_local_flipped, [1])
        bwd_global = torch.flip(bwd_global_flipped, [1])
        merged = torch.cat([fwd_local, fwd_global, bwd_local, bwd_global], dim=-1)
        return self.act(self.layer_norm(self.fusion(merged)))    


# Audio Student
class AudioCfC(nn.Module):
    def __init__(self, num_classes=5):
        super(AudioCfC, self).__init__()
        self.audio_encoder = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 32, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.1, True), nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.1, True), nn.AvgPool1d(3),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.1, True), nn.AvgPool1d(4),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.1, True), nn.AvgPool1d(5),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.1, True))
        
        self.encoder_out_dim = 64
        self.seq_len=16
        self.window_size=4
        self.p_encoder=0.2
        self.p_classifier=0.3
        
        self.encoder_dropout = nn.Dropout(self.p_encoder)
        
        # Bidirectional Parallel Cross-Slice CfC
        self.bi_parallel_cfc = BiParallelCrossSliceCfC(
            input_size=self.encoder_out_dim,
            wiring_units=self.encoder_out_dim * 2,
            output_size=self.encoder_out_dim,
            seq_len=self.seq_len,
            window_size=self.window_size
        )
        
        # Dual-Resolution Pooling
        self.drasp = DRASP(input_dim=self.encoder_out_dim, segment_len=self.window_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_out_dim * 4, 64),
            nn.LayerNorm(64), nn.GELU(), nn.Dropout(self.p_classifier),
            nn.Linear(64, num_classes))

    def forward(self, x):
        # x: (B, 1, 48000)
        x = self.audio_encoder(x) # -> (B, 16, 64)
        x = x.permute(0, 2, 1) # -> (B, 64, 16)
        x = self.encoder_dropout(x)
        seq_features = self.bi_parallel_cfc(x) 
        pooled_features = self.drasp(seq_features.permute(0, 2, 1))
        out = self.classifier(pooled_features)
        return out, seq_features


if __name__ == "__main__":
    model = AudioCfC()
    x = torch.randn(1, 1, 48000)  # (batch, time, features)
    output, hn = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"最终隐藏状态形状: {hn.shape}")
    
    model.eval()
    from ptflops import get_model_complexity_info
    input_res = (1,48000)
    macs, params = get_model_complexity_info(model, input_res, as_strings=True, print_per_layer_stat=False)
    print(f"模型 FLOPs: {macs}")
    print(f"模型参数量: {params}")
    print('#'*80)
    import fvcore
    from fvcore.nn import FlopCountAnalysis,parameter_count_table,parameter_count
    input_res = torch.randn(1,1,48000)
    # 计算 FLOPs
    flops = FlopCountAnalysis(model, input_res)
    print("FLOPs: ", flops.total())
    print(f"FLOPs: {flops.total() / 1e9:.2f} G")
    flops_m = flops.total() / 1e6
    print(f"FLOPs: {flops_m:.2f} M")
    
    # 计算参数量
    params = parameter_count_table(model)
    total_params = parameter_count(model)['']
    params_k = total_params / 1e3
    print(f"Params: {params_k:.2f} K")

    import time
    device = torch.device("cuda:4")
    model.to(device)
    input_tensor = torch.randn(1,1,48000).to(device)
    num_runs = 100
    total_time = 0
    for _ in range(10):
        _ = model(input_tensor)
    torch.cuda.synchronize()  
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_tensor)
    torch.cuda.synchronize()  
    end_time = time.time()
    total_time = end_time - start_time
    average_infer_time = total_time / num_runs
    print(f"Average inference time: {average_infer_time * 1000:.2f} ms")


