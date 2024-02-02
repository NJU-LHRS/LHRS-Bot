import functools
import logging
import os
import socket
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as torch_dist
from torch import Tensor
from torch._C._distributed_c10d import ProcessGroup

logger = logging.getLogger("train")


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()


def get_data_device(data: Union[Tensor, Mapping, Iterable]) -> torch.device:
    """Return the device of ``data``.

    If ``data`` is a sequence of Tensor, all items in ``data`` should have a
    same device type.

    If ``data`` is a dict whose values are Tensor, all values should have a
    same device type.

    Args:
        data (Tensor or Sequence or dict): Inputs to be inferred the device.

    Returns:
        torch.device: The device of ``data``.

    Examples:
        >>> import torch
        >>> from mmengine.dist import cast_data_device
        >>> # data is a Tensor
        >>> data = torch.tensor([0, 1])
        >>> get_data_device(data)
        device(type='cpu')
        >>> # data is a list of Tensor
        >>> data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        >>> get_data_device(data)
        device(type='cpu')
        >>> # data is a dict
        >>> data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([0, 1])}
        >>> get_data_device(data)
        device(type='cpu')
    """
    if isinstance(data, Tensor):
        return data.device
    elif isinstance(data, Mapping):
        pre = None
        for v in data.values():
            cur = get_data_device(v)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        "device type in data should be consistent, but got "
                        f"{cur} and {pre}"
                    )
        if pre is None:
            raise ValueError("data should not be empty.")
        return pre
    elif isinstance(data, Iterable) and not isinstance(data, str):
        pre = None
        for item in data:
            cur = get_data_device(item)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        "device type in data should be consistent, but got "
                        f"{cur} and {pre}"
                    )
        if pre is None:
            raise ValueError("data should not be empty.")
        return pre
    else:
        raise TypeError(
            "data should be a Tensor, sequence of tensor or dict, " f"but got {data}"
        )


def get_backend(group: Optional[ProcessGroup] = None) -> Optional[str]:
    """Return the backend of the given process group.

    Note:
        Calling ``get_backend`` in non-distributed environment will return
        None.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific
            group is specified, the calling process must be part of
            :attr:`group`. Defaults to None.

    Returns:
        str or None: Return the backend of the given process group as a lower
        case string if in distributed environment, otherwise None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_backend(group)
    else:
        return None


def get_comm_device(group: Optional[ProcessGroup] = None) -> torch.device:
    """Return the device for communication among groups.

    Args:
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        torch.device: The device of backend.
    """
    backend = get_backend(group)
    if backend == "hccl":
        import torch_npu  # noqa: F401

        return torch.device("npu", torch.npu.current_device())
    elif backend == torch_dist.Backend.NCCL:
        return torch.device("cuda", torch.cuda.current_device())
    elif backend == "cncl":
        import torch_mlu  # noqa: F401

        return torch.device("mlu", torch.mlu.current_device())
    elif backend == "smddp":
        return torch.device("cuda", torch.cuda.current_device())
    else:
        # GLOO and MPI backends use cpu device by default
        return torch.device("cpu")


def cast_data_device(
    data: Union[Tensor, Mapping, Iterable],
    device: torch.device,
    out: Optional[Union[Tensor, Mapping, Iterable]] = None,
) -> Union[Tensor, Mapping, Iterable]:
    """Recursively convert Tensor in ``data`` to ``device``.

    If ``data`` has already on the ``device``, it will not be casted again.

    Args:
        data (Tensor or list or dict): Inputs to be casted.
        device (torch.device): Destination device type.
        out (Tensor or list or dict, optional): If ``out`` is specified, its
            value will be equal to ``data``. Defaults to None.

    Returns:
        Tensor or list or dict: ``data`` was casted to ``device``.
    """
    if out is not None:
        if type(data) != type(out):
            raise TypeError(
                "out should be the same type with data, but got data is "
                f"{type(data)} and out is {type(data)}"
            )

        if isinstance(out, set):
            raise TypeError("out should not be a set")

    if isinstance(data, Tensor):
        if get_data_device(data) == device:
            data_on_device = data
        else:
            data_on_device = data.to(device)

        if out is not None:
            # modify the value of out inplace
            out.copy_(data_on_device)  # type: ignore

        return data_on_device
    elif isinstance(data, Mapping):
        data_on_device = {}
        if out is not None:
            data_len = len(data)
            out_len = len(out)  # type: ignore
            if data_len != out_len:
                raise ValueError(
                    "length of data and out should be same, "
                    f"but got {data_len} and {out_len}"
                )

            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device, out[k])  # type: ignore
        else:
            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device)

        if len(data_on_device) == 0:
            raise ValueError("data should not be empty")

        # To ensure the type of output as same as input, we use `type(data)`
        # to wrap the output
        return type(data)(data_on_device)  # type: ignore
    elif (
        isinstance(data, Iterable)
        and not isinstance(data, str)
        and not isinstance(data, np.ndarray)
    ):
        data_on_device = []
        if out is not None:
            for v1, v2 in zip(data, out):
                data_on_device.append(cast_data_device(v1, device, v2))
        else:
            for v in data:
                data_on_device.append(cast_data_device(v, device))

        if len(data_on_device) == 0:
            raise ValueError("data should not be empty")

        return type(data)(data_on_device)  # type: ignore
    else:
        raise TypeError(
            "data should be a Tensor, list of tensor or dict, " f"but got {data}"
        )


@functools.lru_cache()
def _get_global_gloo_group() -> ProcessGroup:
    """Return a process group based on gloo backend, containing all ranks.
    The result is cached.
    """
    if torch_dist.get_backend() == "nccl":
        return torch_dist.new_group(backend="gloo")
    else:
        return torch_dist.group.WORLD


def all_gather(data: Tensor, group: Optional[ProcessGroup] = None) -> List[Tensor]:
    """Gather data from the whole group in a list.

    Note:
        Calling ``all_gather`` in non-distributed environment does nothing
        and just returns a list containing :attr:`data` itself.

    Note:
        Unlike PyTorch ``torch.distributed.all_gather``, :meth:`all_gather` in
        MMEngine does not pass in an empty list ``gather_list`` and returns
        the ``gather_list`` directly, which is more convenient. The difference
        between their interfaces is as below:

        - MMEngine: all_gather(data, group) -> gather_list
        - PyTorch: all_gather(gather_list, data, group) -> None

    Args:
        data (Tensor): Tensor to be gathered.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Tensor]: Return a list containing data from the whole group if
        in distributed environment, otherwise a list only containing
        :attr:`data` itself.

    Examples:
        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> output = dist.all_gather(data)
        >>> output
        [tensor([0, 1])]

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2])  # Rank 0
        tensor([3, 4])  # Rank 1
        >>> output = dist.all_gather(data)
        >>> output
        [tensor([1, 2]), tensor([3, 4])]  # Rank 0
        [tensor([1, 2]), tensor([3, 4])]  # Rank 1
    """
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    input_device = get_data_device(data)
    backend_device = get_comm_device(group)
    data_on_device = cast_data_device(data, backend_device)

    gather_list = [
        torch.empty_like(data, device=backend_device) for _ in range(world_size)
    ]

    torch_dist.all_gather(gather_list, data_on_device, group)

    return cast_data_device(gather_list, input_device)  # type: ignore


def gather(data: Any, dst: int = 0, group: Optional[ProcessGroup] = None) -> List[Any]:
    """Run :meth:`gather` on arbitrary picklable data (not necessarily tensors).

    Args:
        data: Any picklable object.
        dst (int): Destination rank.
        group (ProcessGroup): A torch process group. By default, will use a group which
            contains all ranks on ``gloo`` backend.

    Returns:
        list[data]: On ``dst``, a list of data gathered from each rank. Otherwise, an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = torch_dist.get_world_size(group)
    if world_size == 1:
        return [data]

    if torch_dist.get_rank(group) == dst:
        output = [None for _ in range(world_size)]
        torch_dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        torch_dist.gather_object(data, None, dst=dst, group=group)
        return []


def reduce_dict(
    input_dict: Dict[str, Tensor], average: bool = True
) -> Dict[str, Tensor]:
    """Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.

    Args:
        input_dict (dict): All the values will be reduced.
        average (bool): Whether to do average or sum.

    Returns:
        dict: A dict with the same fields as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch_dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def setup_print_for_distributed(is_master: bool) -> None:
    """This function disables printing when not in master process.

    Args:
        is_master (bool): If the current process is the master process or not.
    """
    import builtins

    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """

    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0


def sync_random_seed(group: Optional[ProcessGroup] = None) -> int:
    """Synchronize a random seed to all processes.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Random seed.
    """
    seed = np.random.randint(2**31)
    if get_world_size(group) == 1:
        return seed

    if group is None:
        group = get_default_group()

    backend_device = get_comm_device(group)

    if get_rank(group) == 0:
        random_num = torch.tensor(seed, dtype=torch.int32).to(backend_device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32).to(backend_device)

    torch_dist.broadcast(random_num, src=0, group=group)

    return random_num.item()


def is_main_process() -> bool:
    """Return if the current process is the master process or not."""
    return get_rank() == 0


def _is_free_port(port: int) -> bool:
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append("localhost")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def deepspeed_init_distributed() -> Tuple[int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # launched by `torch.distributed.launch`
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        # launched by slurm
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode.")
        return 0, 0, 1

    print(f"| distributed init (rank {rank})", flush=True)
    deepspeed.init_distributed()
    torch_dist.barrier()
    torch.cuda.set_device(local_rank)
    setup_print_for_distributed(rank == 0)
    return rank, local_rank, world_size


def init_distributed(auto: bool = False) -> Tuple[int]:
    """Initialize the distributed mode as follows:

    - Initialize the process group, with ``backend="nccl"`` and ``init_method="env://"``.
    - Set correct cuda device.
    - Disable printing when not in master process.

    Args:
        auto (bool): If True, when MASTER_PORT is not free, automatically find a free one.
            Defaults to False.

    Returns:
        tuple: (``rank``, ``local_rank``, ``world_size``)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # launched by `torch.distributed.launch`
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        # launched by slurm
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode.")
        return 0, 0, 1

    assert "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ, (
        "init_method='env://' requires the two environment variables: "
        "MASTER_ADDR and MASTER_PORT."
    )

    if auto:
        assert (
            os.environ["MASTER_ADDR"] == "127.0.0.1"
        ), "`auto` is not supported in multi-machine jobs."
        port = os.environ["MASTER_PORT"]
        if not _is_free_port(port):
            new_port = _find_free_port()
            print(f"Port {port} is not free, use port {new_port} instead.")
            os.environ["MASTER_PORT"] = new_port

    print(f"| distributed init (rank {rank})", flush=True)
    torch_dist.init_process_group(backend="nccl")
    torch_dist.barrier()
    torch.cuda.set_device(local_rank)
    setup_print_for_distributed(rank == 0)
    return rank, local_rank, world_size
