import inspect
import logging

from typing import Callable

import torch

from diffusers import DiffusionPipeline
from diffusers import FluxPipeline

from liger_kernel.ops.layer_norm import LigerRMSNorm

logger = logging.getLogger(__name__)
SUPPORTED_DIFFUSERS_VERSION = "0.32.1"
DIFFUSERS_DEPRECATION_WARNING = "Support for diffusers versions < 0.32.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/diffusers/pull/10000"


def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)


def _patch_rms_norm_module(module, offset=0.0, eps=1e-6, casting_mode="llama", in_place=True):
    module.offset = offset
    module.casting_mode = casting_mode
    module.variance_epsilon = getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
    module.in_place = in_place
    _bind_method_to_module(module, "forward", LigerRMSNorm.forward)
    _bind_method_to_module(module, "extra_repr", LigerRMSNorm.extra_repr)


def apply_liger_kernel_to_flux(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    pipeline: DiffusionPipeline = None,
) -> None:
    """Apply Liger kernel optimizations to Flux pipeline."""

    if not isinstance(pipeline, FluxPipeline):
        raise ValueError("Pipeline must be a FluxPipeline")

    # Store original modules for potential restoration
    _original_modules = {}

    if rms_norm:
        for name, module in pipeline.flux_model.named_modules():
            if isinstance(module, torch.nn.RMSNorm):
                _original_modules[name] = module
                # Replace with Liger optimized version
                optimized_module = _patch_rms_norm_module(module)
                module_path = name.split(".")
                current = pipeline
                for part in module_path[:-1]:
                    current = getattr(current, part)
                setattr(current, module_path[-1], optimized_module)

    if swiglu:
        # Implement SwiGLU optimization
        pass


# Map pipeline types to their optimization functions
PIPELINE_TYPE_TO_APPLY_LIGER_FN = {
    "FluxPipeline": apply_liger_kernel_to_flux,
}


def _apply_liger_kernel(pipeline_type: str, **kwargs) -> None:
    """Apply Liger kernels based on pipeline type."""
    if not pipeline_type:
        logger.info("Pipeline type was not provided. No Liger kernels will be applied.")
        return

    if pipeline_type not in PIPELINE_TYPE_TO_APPLY_LIGER_FN:
        logger.info(f"There are currently no Liger kernels supported for pipeline type: {pipeline_type}")
        return

    apply_fn = PIPELINE_TYPE_TO_APPLY_LIGER_FN[pipeline_type]

    # Filter kwargs to only include those accepted by the apply function
    apply_fn_signature = inspect.signature(apply_fn)
    applicable_kwargs = {key: value for key, value in kwargs.items() if key in apply_fn_signature.parameters}

    logger.info(f"Applying Liger kernels for pipeline type: {pipeline_type} with kwargs: {applicable_kwargs}")
    apply_fn(**applicable_kwargs)


def _apply_liger_kernel_to_instance(pipeline: DiffusionPipeline, **kwargs) -> None:
    """Apply Liger kernels to an existing pipeline instance."""
    pipeline_type = pipeline.__class__.__name__
    if pipeline_type not in PIPELINE_TYPE_TO_APPLY_LIGER_FN:
        logger.info(f"No Liger optimizer available for pipeline type: {type(pipeline)}")
        return

    apply_fn = PIPELINE_TYPE_TO_APPLY_LIGER_FN[pipeline_type]
    apply_fn(pipeline=pipeline, **kwargs)
