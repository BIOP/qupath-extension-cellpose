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
import numpy as np
from cellpose import models, io
from typing import TYPE_CHECKING


###############################################################################
# AUXILIARY FUNCTIONS
###############################################################################

def filter_channels(selected_channels: list[int | None]) -> list[int]:
    """Filter out None values from a list of channel indices."""
    merged = [c for c in selected_channels if c is not None]
    if not merged:
        raise ValueError("At least one channel must be provided, only `None` were given.")
    return merged


###############################################################################
# PROCESSING FUNCTIONS
###############################################################################

def run_cellpose_v4(img: np.ndarray, kwargs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs Cellpose v4 on a single image with the given parameters. Refer to Cellpose documentation for kwargs list."""

    model: CellposeModel | None = globals()['model']
    if model is None:
        # Use cpsam pretrained model by default if no custom model is provided
        custom_model = kwargs.get('custom_model', None)
        selected_model = "cpsam" if custom_model is None else custom_model

        task.update(
         current = 2,
         maximum= 5,
         message=f"CP4: Deploy model {selected_model if selected_model else custom_model}"
        )
        model = models.CellposeModel(
            pretrained_model=selected_model,
            gpu=kwargs.get('use_gpu', False),
            device=kwargs.get('device', None)
        )
        task.update(
          current = 3,
          maximum= 5,
          message=f"CP4: Predict labels (device={device})"
       )

    channel_axis = kwargs.get('channel_axis', None)
    z_axis = kwargs.get('z_axis', None)
    time_axis = kwargs.get('time_axis', None)
    stitch_threshold=kwargs.get('stitch_threshold', 0.)
    do_3D=kwargs.get('use_3D', False)

    task.update(message=f"Received image with shape {img.shape} and parameters: channel_axis={channel_axis}, z_axis={z_axis}, time_axis={time_axis}, stitch_threshold={stitch_threshold}, use_3D={do_3D}")

    # Cellpose 4 madness to get batch processing for 2D+T or 2D+C+T images
    if time_axis is not None and z_axis is None:
        # Cellpose 2D batch mode: process each T frame independently, no stitching.
        stitch_threshold = 0.
        do_3D = False
        # convert_image only handles channel_axis for ndim==3 (one batch+spatial+channel).
        # For 4D [T, C, Y, X] we must reorder to [T, Y, X, C] so Cellpose's ndim==4 path
        # (which expects channels-last) takes over without needing channel_axis.
        if img.ndim == 4 and channel_axis is not None:
            t = time_axis % img.ndim
            c = channel_axis % img.ndim
            spatial = [ax for ax in range(img.ndim) if ax not in (t, c)]
            img = np.transpose(img, (t, spatial[0], spatial[1], c))
            channel_axis = None  # channels are now last; let Cellpose detect them
        else:
            # We have no channel axis. add a fake one so Cellpose doesn't complain, and remove it after prediction.
            img = np.expand_dims(img, axis=-1)
            channel_axis = None
    
    # Another special case. If we have a Z stack, but stitch_threshold= 0., cellpose will
    # complain that the z_axis should be None to process a batch of 2D planes. We abide.
    # But then we must also check that we have a channel image.
    if not do_3D and z_axis is not None and stitch_threshold == 0.:
        z_axis = None
        if channel_axis is None:
            img = np.expand_dims(img, axis=-1)
        else:
            # Move channel_axis to the end.
            c = channel_axis % img.ndim
            spatial = [ax for ax in range(img.ndim) if ax != c]
            img = np.transpose(img, (*spatial, c))
            channel_axis = None

    task.update(message=f"Final parameters for model.eval: channel_axis={channel_axis}, z_axis={z_axis}, stitch_threshold={stitch_threshold}, do_3D={do_3D}. Image shape: {img.shape}")
    
    masks, flows, styles = model.eval(
        img,
        diameter=kwargs.get('diameter', 30),
        do_3D=do_3D,
        anisotropy=kwargs.get('anisotropy', 1.0),
        stitch_threshold=stitch_threshold,
        channel_axis=channel_axis,
        z_axis=z_axis,
        flow3D_smooth=kwargs.get('flow3D_smooth', 0),
        resample=kwargs.get('resample', True),
        normalize=kwargs.get('normalize', True),
        flow_threshold=kwargs.get('flow_threshold', 0.4),
        cellprob_threshold=kwargs.get('cellprob_threshold', 0.0),
        min_size=kwargs.get('min_size', 15),
        niter=kwargs.get( 'niter', None ),
        tile_overlap=kwargs.get('tile_overlap', 0.1),
    )

    task.update(message=f"Model evaluation completed. Masks shape: {masks.shape}, Flows shape: {flows[0].shape}, Styles shape: {styles.shape}")

    return masks, flows, styles


###############################################################################
# MAIN PROGRAM
###############################################################################



appose_mode = 'task' in globals()
if appose_mode:
    if TYPE_CHECKING:
        from appose.python_worker import Task
        task: Task

    from appose.python_worker import Task
    task = globals()['task']
else:
    from cp_utils import get_torch_device
    from appose.python_worker import Task
    task = Task()

# load images
if appose_mode:
    fiji_image = globals()['input']
    fiji_output_labels = globals()['output_labels']
    fiji_output_flows = globals()['output_flows']

    stitch_threshold = globals()['stitch_threshold']
    z_axis: int | None = globals()['z_axis']
    channel_axis: int | None = globals()['channel_axis']
    time_axis: int | None = globals()['t_axis']
    anisotropy: float = globals()['anisotropy']
    diameter: int = globals()['diameter']
    use_3D: bool = globals()['use_3D']
    resample: bool = globals()['resample']
    normalize: bool = globals()['normalize']
    flow_threshold: float = globals()['flow_threshold']
    cellprob_threshold: float = globals()['cellprob_threshold']
    min_size: int = globals()['min_size']
    niter: int | None = globals()['niter']
    tile_overlap: float = globals()['tile_overlap']
    flow3D_smooth: float = globals()['flow3D_smooth']
    n_channels: int = globals()['n_channels']    
    use_gpu: bool = globals()['use_gpu']

    
    input_image = fiji_image.ndarray()  # pylint: disable=E1120
    output_labels = fiji_output_labels.ndarray()
    output_flows = fiji_output_flows.ndarray() if fiji_output_flows is not None else None    
    anisotropy = anisotropy if anisotropy > 0 else None
    
    if channel_axis is not None:
        chan0: int | None = globals()['chan0']
        chan1: int | None = globals()['chan1']
        chan2: int | None = globals()['chan2']
        channels = filter_channels([chan0, chan1, chan2])
        if len(input_image.shape) > 2 :
            input_image = input_image[..., channels, :, :]

    task.update(
        current = 0,
        maximum = 5,
        message = f"CP4: Fetch image from Fiji ({input_image.shape})"
        )
else:
    import os
    sample_folder = '../../../samples/' # When you run this script from its location.
    
    test_file = 'testImg_XYCT.tif'
    z_axis = None
    channel_axis = 1
    time_axis = 0

    file = os.path.join(sample_folder, test_file) 
    input_image = io.imread(file)
    custom_model = None
    diameter = 30
    use_3D = False
    stitch_threshold = 0.
    anisotropy = None
    compute_flows = True
    resample = True
    normalize = True
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    min_size = 15
    niter = None
    tile_overlap = 0.1
    flow3D_smooth = 0
    use_gpu = False

use_gpu, device = get_torch_device(use_gpu)
task.update(
    current = 1,
    maximum= 5,
    message=f"CP4: Start Cellpose (device={device})"
)

masks, flows, styles = run_cellpose_v4(
    input_image,
    kwargs={
        "diameter": diameter,
        "use_3D": use_3D,
        "stitch_threshold": stitch_threshold,
        "anisotropy": anisotropy,
        "channel_axis": channel_axis,
        "z_axis": z_axis,
        "time_axis": time_axis,
        "use_gpu": use_gpu,
        "device": device,
        'flow3D_smooth': flow3D_smooth,
        'resample': resample,
        'normalize': normalize,
        'flow_threshold': flow_threshold,
        'cellprob_threshold': cellprob_threshold,
        'min_size': min_size,
        'niter': niter,
        'tile_overlap': tile_overlap,
    }
)

task.update(
    current = 4,
    maximum = 5,
    message=f"CP4: Returning results"
)

# task.update(message=f'Flows shape before flip: {flows[0].shape if compute_flows else None}')

# Massage outputs
if compute_flows:
    # Move the last axis (C axis) to before Y and X. There might other dims before.
    flows = np.moveaxis(flows[0], -1, -3) if compute_flows else None

# return output
# task.update(message=f'Input image shape: {input_image.shape}')
# task.update(message=f'Masks shape: {masks.shape}')
# task.update(message=f'Flows shape: {flows.shape if compute_flows else None}')
# task.update(message=f'Z axis: {z_axis}, Time axis: {time_axis}')

if appose_mode:
    # Write masks into the shared output image.
    output_labels[:] = masks
    if compute_flows:
        output_flows[:] = flows
else:
    save_path = os.path.join(sample_folder, test_file.replace('.tif', '_masks.tif'))
    io.imsave(save_path, masks.astype(np.uint16))
    if compute_flows:
        flow_save_path = os.path.join(sample_folder, test_file.replace('.tif', '_flows.tif'))
        io.imsave(flow_save_path, flows.astype(np.float32))

task.update(
    current = 5,
    maximum = 5,
    message=f"CP4: Cellpose processing completed"
)
