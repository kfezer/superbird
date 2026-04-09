from truss_train import (
    TrainingProject,
    TrainingJob,
    Image,
    Compute,
    Runtime,
    CacheConfig,
    CheckpointingConfig,
)
from truss.base.truss_config import AcceleratorSpec

# pytorch/pytorch base image ships with torch + CUDA pre-installed.
# train.sh only needs to pip install the remaining deps (torchvision, Pillow, tqdm).
BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = Runtime(
    start_commands=[
        "chmod +x ./train.sh && ./train.sh",
    ],
    cache_config=CacheConfig(enabled=True),
    checkpointing_config=CheckpointingConfig(
        enabled=True,
        volume_size_gib=20,  # MobileNetV3-Small checkpoints are ~25 MB each; 20 GiB is generous
    ),
)

# A10G (24 GB VRAM) is plenty for MobileNetV3-Small on CUB-200 (~11k images).
# Swap to H100 if you want to sweep hyperparams or scale up to EfficientNet-B4+.
training_compute = Compute(
    accelerator=AcceleratorSpec(accelerator="A10G", count=1),
)

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = TrainingProject(
    name="superbird-cub200",
    job=training_job,
)
