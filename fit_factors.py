import argparse
import logging
from datetime import timedelta

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import default_data_collator

from pipeline import construct_olmo2, get_olmino_dataset
from task import LanguageModelingTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.factor_arguments import (
    extreme_reduce_memory_factor_arguments,
)
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Influence factor computation on Olmino dataset.")

    parser.add_argument(
        "--factors_name",
        type=str,
        required=True,
        help="Name of the factor.",
    )
    parser.add_argument(
        "--model_size", 
        type=str, 
        required=True,
        choices=["1B", "7B", "13B", "32B"]
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--factor_batch_size",
        type=int,
        default=4,
        help="Batch size for computing influence factors.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    parser.add_argument(
        "--data_partitions",
        type=int,
        default=1,
        help="Data partitions for both covariance and lambda."
    )
    parser.add_argument(
        "--covariance_model_partitions",
        type=int,
        default=8,
        help="Module partitions for covariance."
    )
    parser.add_argument(
        "--lambda_model_partitions",
        type=int,
        default=8,
        help="Module partitions for lambda."
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_olmino_dataset(model_size=args.model_size)

    # Prepare the trained model.
    model = construct_olmo2(model_size=args.model_size)

    # Define task and prepare model.
    task = LanguageModelingTask(model_size=args.model_size)
    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours.
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    full_output_dir = f"/capstor/scratch/cscs/laylaylo/if_multilingual/kronfluence/exp1/OLMo-2-{args.model_size}-Instruct"

    analyzer = Analyzer(
        analysis_name=f"olmino-mix-run1-15k",
        model=model,
        task=task,
        profile=args.profile,
        output_dir=full_output_dir
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(num_workers=4, collate_fn=default_data_collator, pin_memory=True)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    factors_name = args.factors_name
    factor_args = extreme_reduce_memory_factor_arguments(
        strategy=args.factor_strategy, module_partitions=1, dtype=torch.bfloat16
    )
    factor_args.covariance_module_partitions = args.covariance_model_partitions
    factor_args.lambda_module_partitions = args.lambda_model_partitions
    factor_args.covariance_data_partitions = args.data_partitions
    factor_args.lambda_data_partitions = args.data_partitions
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )


if __name__ == "__main__":
    main()
