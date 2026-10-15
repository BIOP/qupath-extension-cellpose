###
# #%L
# Running Cellpose 3 and 4 from Java with Appose, using ImgLib2 data structure.
# %%
# Copyright (C) 2026 Appose developpers
# %%
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the ImgLib2 nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
# #L%
###

# These imports are required for Appose calls to work on Windows platforms.
import numpy as np
from cellpose import models

import torch

def get_torch_device(use_gpu: bool) -> tuple[bool, torch.device]:
    """Check torch device availability and returns a tupple (use_gpu: bool, device: torch.device) using the best available backend: CUDA > MPS > CPU."""
    if not use_gpu:
        return False, torch.device("cpu")

    if torch.cuda.is_available():
        return True, torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return True, torch.device("mps")

    return False, torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """The device plus the hardware behind it, e.g. ``cuda (NVIDIA GeForce RTX 2080 Ti)``."""
    try:
        if device.type == "cuda":
            return f"{device} ({torch.cuda.get_device_name(device)})"
        if device.type == "mps":
            return f"{device} (Apple Metal)"
    except Exception:  # noqa: BLE001 -- a name lookup must never break the model load
        pass
    return str(device)


def runtime_versions() -> str:
    """One line naming the interpreter and the packages that decide how this worker behaves.

    Logged once per worker start, so a bug report says which cellpose, torch and appose-python
    the worker actually ran (the committed lock decides them, not the extension version).
    """
    import sys
    from importlib import metadata

    def version_of(name: str) -> str:
        try:
            return metadata.version(name)
        except Exception:  # noqa: BLE001 -- an unknown version is reported, not raised
            return "unknown"

    return (
        f"python {sys.version.split()[0]}, cellpose {version_of('cellpose')}, "
        f"torch {torch.__version__}, appose {version_of('appose')}"
    )

# %%
