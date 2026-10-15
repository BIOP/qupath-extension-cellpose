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
from cellpose import models
from typing import TYPE_CHECKING

report = print

def listen(callback):
    global report
    report = callback


appose_mode = 'task' in globals()
if appose_mode:
    if TYPE_CHECKING:
        from appose.python_worker import Task
        task: Task

    from appose.python_worker import Task
    task = globals()['task']
    listen(task.update)
else:
    from cp_utils import get_torch_device
    from appose.python_worker import Task
    import os
    sample_folder = '../../../samples/' # When you run this script from its location.
    task = Task()

# load arguments and input from Appose task
if appose_mode:
    use_gpu: bool = globals()['use_gpu']
    selected_model = model_name if custom_model is None else None
else:
    custom_model = None
    model_name = 'cyto3'
    use_gpu = False

use_gpu, device = get_torch_device(use_gpu)

task.update(
    current = 1,
    maximum= 2,
    message=f"CP3: Start Cellpose (device={describe_device(device)}; {runtime_versions()}): deploy model {selected_model if selected_model else custom_model}"
)

model = models.CellposeModel(
        model_type=selected_model,
        pretrained_model=custom_model,
        gpu=use_gpu,
        device=device
 )

task.update(
   current = 2,
   maximum= 2,
   message=f"CP3: Model initialized"
)

if appose_mode:
	task.export( model=model )
# %%
 
