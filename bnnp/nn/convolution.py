"""
Convs with Muon support.
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple
from torch.nn.parameter import Parameter


class _ConvNd(nn.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:  # type: ignore[empty-body]
        ...

    in_channels: int
    _reversed_padding_repeated_twice: list[int]
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: Union[str, tuple[int, ...]]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: Union[str, tuple[int, ...]],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if transposed:
            # self.weight = Parameter(
            #     torch.empty(
            #         (in_channels, out_channels // groups, *kernel_size),
            #         **factory_kwargs,
            #     )
            # )
            raise NotImplementedError

        else:
            # self.weight = Parameter(
            #     torch.empty(
            #         (out_channels, in_channels // groups, *kernel_size),
            #         **factory_kwargs,
            #     )
            # )
            self.weight = Parameter(
                torch.empty(
                    (out_channels, in_channels // groups * math.prod(kernel_size)),
                    **factory_kwargs,
                )
            )
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573

        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     if fan_in != 0:
        #         bound = 1 / math.sqrt(fan_in)
        #         init.uniform_(self.bias, -bound, bound)

        init.normal_(self.weight, std=0.5 / math.sqrt(self.weight.size(1)))
        if self.bias is not None:
            init.zeros_(self.bias)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight.reshape(
            self.out_channels, self.in_channels // self.groups, *self.kernel_size
        )

        return self._conv_forward(input, weight, self.bias)


class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight.reshape(
            self.out_channels, self.in_channels // self.groups, *self.kernel_size
        )
        return self._conv_forward(input, weight, self.bias)


class Conv3d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight.reshape(
            self.out_channels, self.in_channels // self.groups, *self.kernel_size
        )
        return self._conv_forward(input, weight, self.bias)
