import datetime
import json
import logging
import os
import time
from collections import defaultdict, deque
from enum import Enum
from functools import partial
from typing import Dict, Optional

import ml_collections
import torch
import torch.nn as nn
import wandb
from torch.nn.functional import one_hot, softmax
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from ..utils import get_rank, get_world_size, is_distributed, is_main_process
from .hookbase import HookBase
from .logger_hook import LoggerHook

logger = logging.getLogger("train")


class KnnEvaluate(HookBase):
    def __init__(self, config: ml_collections.ConfigDict):
        super(KnnEvaluate, self).__init__()
        self.nb = config.k
        self.temp = config.T
        self.iter = config.n_iter
        self.enable_amp = config.enable_amp
        self.gather_on_cpu = config.gather_on_cpu

    def after_epoch(self) -> None:
        self.trainer.model_or_module.eval()
        train_loader, test_loader = self.trainer.eval_data_loader

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            results_dict_knn = eval_knn(
                model=self.trainer.model,
                train_loader=train_loader,
                val_loader=test_loader,
                accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
                nb_knn=self.nb,
                temperature=self.temp,
                gather_on_cpu=self.gather_on_cpu,
                n_tries=self.iter,
                device=self.trainer.device,
            )

        results_dict = {}
        if is_main_process():
            for knn_ in results_dict_knn.keys():
                top1 = results_dict_knn[knn_]["top-1"].item() * 100.0
                top5 = results_dict_knn[knn_]["top-5"].item() * 100.0
                results_dict[f"{knn_} Top 1"] = top1
                results_dict[f"{knn_} Top 5"] = top5
                logger.info(
                    f"{knn_} classifier result: Top1: {top1:.2f} Top5: {top5:.2f}"
                )

            for hooks in self.trainer._hooks:
                if isinstance(hooks, LoggerHook):
                    _tb_writer = hooks._tb_writer
                    use_wandb = hooks.wandb

            for metric, score in results_dict.items():
                logger.info(f"{metric}: {score:.2f}")
                _tb_writer.add_scalar(f"eval/{metric}", score, self.trainer.epoch)
                if use_wandb:
                    wandb.log({metric: score})

        metrics_file_path = os.path.join(self.trainer.work_dir, "results_eval_knn.json")
        with open(metrics_file_path, "a") as f:
            for k, v in results_dict.items():
                f.write(json.dumps({k: v}) + "\n")

        if is_distributed():
            torch.distributed.barrier()


class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(
        self, train_features, train_labels, nb_knn, T, device, num_classes=1000
    ):
        super().__init__()

        self.global_rank = get_rank()
        self.global_size = get_world_size()

        self.device = device
        self.train_features_rank_T = train_features.chunk(self.global_size)[
            self.global_rank
        ].T.to(self.device)
        self.candidates = (
            train_labels.chunk(self.global_size)[self.global_rank]
            .view(1, -1)
            .to(self.device)
        )

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        if not is_distributed():
            similarity_rank = torch.mm(features_rank, self.train_features_rank_T)
        else:
            broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
            torch.distributed.broadcast(broadcast_shape, source_rank)

            broadcasted = features_rank
            if self.global_rank != source_rank:
                broadcasted = torch.zeros(
                    *broadcast_shape, dtype=features_rank.dtype, device=self.device
                )
            torch.distributed.broadcast(broadcasted, source_rank)

            # Compute the neighbors for `source_rank` among `train_features_rank_T`
            similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)

        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [
                torch.zeros_like(topk_sims) for _ in range(self.global_size)
            ]
            retrieved_rank = [
                torch.zeros_like(neighbors_labels) for _ in range(self.global_size)
            ]

        torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)

        if self.global_rank == target_rank:
            # Perform a second top-k on the k * global_size retrieved neighbors
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            return results
        return None

    def compute_neighbors(self, features_rank):
        if not is_distributed():
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, 0)
            return topk_sims, neighbors_labels

        for rank in range(self.global_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            one_hot(neighbors_labels, num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, samples, device):
        return nn.functional.normalize(
            self.avgpool(self.model.eval_forward(samples, device)).squeeze(), dim=1, p=2
        )


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        if not is_distributed():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", output_file=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")
        pass

    def log_every(
        self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0
    ):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(
                    iteration=i, iter_time=iter_time.avg, data_time=data_time.avg
                )
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / n_iterations
            )
        )


class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value


def extract_features(
    model, data_loader, gather_on_cpu=False, device=torch.device("cpu")
):
    sample_count = len(data_loader.dataset)
    return extract_features_with_dataloader(
        model, data_loader, sample_count, gather_on_cpu, device
    )


def all_gather_and_flatten(tensor_rank):
    if not is_distributed():
        return tensor_rank

    tensor_all_ranks = torch.empty(
        get_world_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


@torch.inference_mode()
def extract_features_with_dataloader(
    model, data_loader, sample_count, gather_on_cpu=False, device=torch.device("cpu")
):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, labels_rank, index in metric_logger.log_every(data_loader, 10):
        labels_rank = labels_rank.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)
        features_rank = model(samples, device).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(
                sample_count, features_rank.shape[-1], device=gather_device
            )
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels


def build_topk_accuracy_metric(
    average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5)
):
    metrics: Dict[str, Metric] = {
        f"top-{k}": MulticlassAccuracy(
            top_k=k, num_classes=int(num_classes), average=average_type.value
        )
        for k in ks
    }
    return MetricCollection(metrics)


def create_class_indices_mapping(labels):
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    mapping = {
        unique_labels[i]: (inverse == i).nonzero() for i in range(len(unique_labels))
    }
    return mapping


class ModuleDictWithForward(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs):
        return {k: module(*args, **kwargs) for k, module in self._modules.items()}


def filter_train(mapping, n_per_class, seed):
    torch.manual_seed(seed)
    final_indices = []
    for k in mapping.keys():
        index = torch.randperm(len(mapping[k]))[:n_per_class]
        final_indices.append(mapping[k][index])
    return torch.cat(final_indices).squeeze()


def create_module_dict(
    *, module, n_per_class_list, n_tries, nb_knn, train_features, train_labels
):
    modules = {}
    mapping = create_class_indices_mapping(train_labels)
    for npc in n_per_class_list:
        if npc < 0:  # Only one try needed when using the full data
            full_module = module(
                train_features=train_features,
                train_labels=train_labels,
                nb_knn=nb_knn,
            )
            modules["full"] = ModuleDictWithForward({"1": full_module})
            continue
        all_tries = {}
        for t in range(n_tries):
            final_indices = filter_train(mapping, npc, seed=t)
            k_list = list(set(nb_knn + [npc]))
            k_list = sorted([el for el in k_list if el <= npc])
            all_tries[str(t)] = module(
                train_features=train_features[final_indices],
                train_labels=train_labels[final_indices],
                nb_knn=k_list,
            )
        modules[f"{npc} per class"] = ModuleDictWithForward(all_tries)

    return ModuleDictWithForward(modules)


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        return {"preds": features_dict, "target": targets}


def eval_knn(
    model,
    train_loader,
    val_loader,
    accuracy_averaging,
    nb_knn,
    temperature,
    gather_on_cpu,
    n_per_class_list=[-1],
    n_tries=1,
    device=torch.device("cpu"),
):
    model = ModelWithNormalize(model)

    logger.info("Extracting features for train set...")
    train_features, train_labels = extract_features(
        model, train_loader, gather_on_cpu=gather_on_cpu, device=device
    )
    logger.info(f"Train features created, shape {train_features.shape}.")

    num_classes = train_labels.max() + 1
    metric_collection = build_topk_accuracy_metric(
        accuracy_averaging, num_classes=num_classes
    )

    partial_module = partial(
        KnnModule, T=temperature, device=device, num_classes=num_classes
    )
    knn_module_dict = create_module_dict(
        module=partial_module,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
        nb_knn=nb_knn,
        train_features=train_features,
        train_labels=train_labels,
    )
    postprocessors, metrics = {}, {}
    for n_per_class, knn_module in knn_module_dict.items():
        for t, knn_try in knn_module.items():
            postprocessors = {
                **postprocessors,
                **{
                    (n_per_class, t, k): DictKeysModule([n_per_class, t, k])
                    for k in knn_try.nb_knn
                },
            }
            metrics = {
                **metrics,
                **{
                    (n_per_class, t, k): metric_collection.clone()
                    for k in knn_try.nb_knn
                },
            }
    model_with_knn = torch.nn.ModuleList()
    model_with_knn.append(model)
    model_with_knn.append(knn_module_dict)

    # ============ evaluation ... ============
    logger.info("Start the k-NN classification.")
    _, results_dict = evaluate(
        model_with_knn, val_loader, postprocessors, metrics, device
    )

    # Averaging the results over the n tries for each value of n_per_class
    for n_per_class, knn_module in knn_module_dict.items():
        first_try = list(knn_module.keys())[0]
        k_list = knn_module[first_try].nb_knn
        for k in k_list:
            keys = results_dict[
                (n_per_class, first_try, k)
            ].keys()  # keys are e.g. `top-1` and `top-5`
            results_dict[(n_per_class, k)] = {
                key: torch.mean(
                    torch.stack(
                        [
                            results_dict[(n_per_class, t, k)][key]
                            for t in knn_module.keys()
                        ]
                    )
                )
                for key in keys
            }
            for t in knn_module.keys():
                del results_dict[(n_per_class, t, k)]

    return results_dict


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        outputs = model[0](samples, device=device)
        outputs = model[1](outputs)
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {
        k: meter.global_avg for k, meter in metric_logger.meters.items()
    }
    return metric_logger_stats, stats
