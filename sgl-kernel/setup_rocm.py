# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import platform
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"
include_dirs = [
    root / "include",
    root / "include" / "impl",
    root / "csrc",
]

cxx_flags = ["-O3"]
libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]

CDNA_TARGETS = {"gfx942", "gfx950"}
RDNA_TARGETS = {"gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201"}
ALL_TARGETS = CDNA_TARGETS | RDNA_TARGETS

RDNA_FP8_TARGETS = {"gfx1200", "gfx1201"}

default_target = "gfx942"
amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)

if torch.cuda.is_available():
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.cuda not available. Using default target: {amdgpu_target}")

if amdgpu_target not in ALL_TARGETS:
    print(
        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. "
        f"Supported: {sorted(ALL_TARGETS)}"
    )
    sys.exit(1)

is_rdna = amdgpu_target in RDNA_TARGETS
has_fp8 = amdgpu_target in CDNA_TARGETS or amdgpu_target in RDNA_FP8_TARGETS

sources = [
    "csrc/allreduce/custom_all_reduce.hip",
    "csrc/allreduce/deterministic_all_reduce.hip",
    "csrc/common_extension_rocm.cc",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/deepseek_v4_topk.cu",
    "csrc/elementwise/dsv4_norm_rope.cu",
    "csrc/elementwise/topk.cu",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/speculative/eagle_utils.cu",
    "csrc/kvcacheio/transfer.cu",
    "csrc/memory/weak_ref_tensor.cpp",
    "csrc/elementwise/pos_enc.cu",
]

if is_rdna:
    sources.append("csrc/allreduce/quick_all_reduce_stub.cc")
else:
    sources.append("csrc/allreduce/quick_all_reduce.cu")

if amdgpu_target == "gfx942":
    fp8_macro = "-DHIP_FP8_TYPE_FNUZ"
elif has_fp8:
    fp8_macro = "-DHIP_FP8_TYPE_E4M3"
else:
    fp8_macro = None

if is_rdna:
    print(f"Note: Building for RDNA target {amdgpu_target} (experimental)")
    if not has_fp8:
        print(f"Note: FP8 disabled for {amdgpu_target} (no native FP8 support)")

# gfx942: 64KB LDS -> 48KB dynamic, gfx95x: 160KB LDS -> 128KB dynamic
# RDNA3/4 (gfx11xx/gfx12xx): 64KB LDS per WGP -> 48KB (matches gfx942 budget)
if amdgpu_target.startswith("gfx95"):
    topk_dynamic_smem_bytes = 32 * 1024 * 4
else:
    topk_dynamic_smem_bytes = 48 * 1024

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    f"--amdgpu-target={amdgpu_target}",
    "-DENABLE_BF16",
    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
]

if has_fp8:
    hipcc_flags.extend(["-DENABLE_FP8", fp8_macro])

if is_rdna:
    hipcc_flags.append("-DSGL_IS_RDNA")
    cxx_flags.append("-DSGL_IS_RDNA")

_rocm_warp = 64 if not is_rdna else 32
hipcc_flags.append(f"-DSGL_ROCM_WARP_SIZE={_rocm_warp}")
cxx_flags.append(f"-DSGL_ROCM_WARP_SIZE={_rocm_warp}")

ext_modules = [
    CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": hipcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
