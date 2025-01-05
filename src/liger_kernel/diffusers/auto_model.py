from diffusers import DiffusionPipeline

from liger_kernel.diffusers.monkey_patch import _apply_liger_kernel

__all__ = ["AutoLigerKernelForDiffusionPipeline"]


class AutoLigerKernelForDiffusionPipeline:
    """Drop-in replacement for diffusers pipeline loading that applies Liger kernel optimizations."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        pipeline_type = pipeline.__class__.__name__
        _apply_liger_kernel(pipeline_type, pipeline=pipeline)
        return pipeline
