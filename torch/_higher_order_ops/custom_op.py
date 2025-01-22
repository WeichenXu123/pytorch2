from torch._ops import HigherOrderOperator
import torch
import numpy as np
from torch._C import DispatchKey
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    ProxyTorchDispatchMode,
    track_tensor_tree
)
import torch.utils._pytree as pytree


class MyOp1(HigherOrderOperator):
    def __init__(self):
        super().__init__("MyOp1")

    def __call__(self, operand):
        return super().__call__(operand)

my_op1_op = MyOp1()



def my_op1(x: torch.Tensor) -> torch.Tensor:
    return my_op1_op(x)


"""
@my_op1_op.py_impl(DispatchKey.CPU)
def my_op1_cpu(x):
    x_np = x.numpy()
    y_np = np.sin(x_np)
    return torch.from_numpy(y_np).to(device=x.device)
"""


class MyOp1AutogradCPUOp(torch.autograd.Function):
    
    @staticmethod
    def forward(x):
        # with torch._C._AutoDispatchBelowAutograd():
        #   return my_op1_op(x)
        with torch._C._DisableTorchDispatch():
            x_np = x.numpy()
            y_np = np.sin(x_np)
            return torch.from_numpy(y_np).to(device=x.device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_out):
        with torch._C._DisableTorchDispatch():
            x, = ctx.saved_tensors
            x_np = x.numpy()
            grad_out_np = grad_out.numpy()
            return torch.from_numpy(grad_out_np * np.cos(x_np))


@my_op1_op.py_impl(DispatchKey.AutogradCPU)
def my_op1_cpu_autograd(x):
    return MyOp1AutogradCPUOp.apply(x)



@my_op1_op.py_impl(FakeTensorMode)
def my_op1_fake_mode(x):
    return x.new_empty(x.shape)


@my_op1_op.py_functionalize_impl
def my_op1_functionalize(ctx, x):
    unwrapped_x = ctx.unwrap_tensors(x)
    with ctx.redispatch_to_next():
        out = ctx.functionalize(my_op1_op)(unwrapped_x)
        return ctx.wrap_tensors(out)
    

@my_op1_op.py_impl(ProxyTorchDispatchMode)
def my_op1_trace(proxy_mode, x):
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, x)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", my_op1_op, proxy_args, {}
    )
    out = my_op1_op(x.shape)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
