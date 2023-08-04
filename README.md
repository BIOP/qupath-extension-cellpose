# QuPath Cellpose/Omnipose extension

This repo adds some support to use 2D Cellpose within QuPath through a Python virtual environment.

We also want to use Omnipose, which offers some amazing features, but there is currently no consensus between the 
developers and there are incompatibilities between the current Cellpose and Omnipose versions.

We have decided to provide support for both using cellpose and omnipose, in the form of two separate environments, so that
they can play nice. 

> **Warning**
> Versions above v0.6.0 of this extension **will only work on QuPath 0.4.0 or later**. Please update QuPath to the latest version. 

> **Warning**
> In case you are stuck with QuPath v0.3.2, [the last release to work is v0.5.1](https://github.com/BIOP/qupath-extension-cellpose/releases/tag/v0.5.1)

# Installation

## Step 1: Install Cellpose and Omnipose

Follow the instructions to install Cellpose from [the main Cellpose repository](https://github.com/mouseland/cellpose). And
Omnipose from [the main Omnipose repository](https://omnipose.readthedocs.io/installation.html)
This extension will need to know the path to at least your Cellpose environment. If you plan on using Omnipose, you will also need to install it. 

### NOTE: `scikit-image` Dependency
As of version 0.4 of this extension, QC (quality control) is run **automatically** when training a model.

Due to the dependencies of the validation code, located inside [run-cellpose-qc.py](QC/run-cellpose-qc.py) requires an 
extra dependency on `scikit-image`.

The simplest way to add it, is when installing Cellpose as instructed in the official repository and adding scikit-image
```bash
python -m pip install cellpose scikit-image
```
or
```bash
python -m pip install omnipose scikit-image
```

## Installation with Conda/Mamba
We provide the following YAML file that installs Cellpose and omnipose in the same environment.
The configuration files are without guarantee, but these are the ones we use for our Windows machines.
[Download `cellpose-omnipose-biop-gpu.yml`](files/cellpose-omnipose-biop-gpu.yml)

You can create the environment with the following command using either conda or mamba:

```bash
mamba env create -f cellpose-omnipose-biop-gpu.yml
```

### Check the path to the Python executable
We will need this information later when configuring the QuPath Cellpose extension.

```
mamba activate cellpose-omnipose-biop-gpu
where python
F:\conda-envs\cellpose-omnipose-biop-gpu\python.exe
```

> **Note**
> While this example is done under Windows, this will work on Mac and Linux as well. 

## Step 2: Install the QuPath Cellpose extension

Download the latest `qupath-extension-cellpose-[version].zip` file from [releases](https://github.com/biop/qupath-extension-cellpose/releases) and unzip it into your `extensions` directory. 

If your extensions directory is unset, unzip and drag & drop `qupath-extension-cellpose-[version].jar` onto the main QuPath window. You'll be prompted to select a QuPath user directory.
The extension will then be copied to a location inside that directory.

To copy `run-cellpose-qc.py`, go to Extensions > Installed Extensions and click on "Open Extensions Directory". You can place the `run-cellpose-qc.py` in the same folder.

You might then need to restart QuPath (but not your computer).

> **Note**
> In case you do not do this step, Cellpose training will still work, but the QC step will be skipped, and you will be notified that `run-cellpose-qc.py` cannot be found.


## QuPath Extension Cellpose/Omnipose: First time setup

Go to `Edit > Preferences > Cellpose/Omnipose`
Complete the fields with the requested information. based on the `mamba` installation above, this is what it should look like:
![Cellpose setup example](files/cellpose-qupath-setup-example.png)
> **Note**
You have the possibility to provide **two** different environments. One for Cellpose and one for Omnipose. 
> If you do not plan on using Omnipose or have installed both cellpose and Omnipose in the same environment, you can leave it blank.

The reason for this is that there may be versions of cellpose and its dependencies that might not match with Omnipose. Adding to that, some parameters
in cellpose and omnipose are currently out of sync, so it could be wiser to keep them separate.

> **Warning** as of this writing, the versions used are `cellpose==2.2.1` and `omnipose==0.4.4`

**The extension handles switching between the two based on the `useOmnipose()` flag in the builder.**

## Running Cellpose the first time in standalone

Cellpose needs to download the pretrained models the first time it is run. On some OSes, this does not work from within 
QuPath due to permission issues.

One trick is to **run Cellpose from the command line** once with the model you want to use. The download should work from there,
and you can then use it within the QuPath Extension Cellpose.

# Using the Cellpose QuPath Extension

## Training

**Requirements**:
A QuPath project with rectangles of class "Training" and "Validation" inside which the ground truth objects have been painted as annotations with no class.
![Example Annotations for Training](files/cellpose-qupath-training-example.png)

We typically create a standalone QuPath project for training only. This project will contain the training images along with the ground truth annotations drawn in QuPath.
Here are some reasons we do it this way:

1. Separating training and prediction/analysis makes for clean project structures and easier sharing of the different steps of your workflow.
2. In case we need to get more ground truth, we can simply fire up the relevant QuPath project and rerun the training, and then import the newly trained model into any other project that might need it.

**Protocol**

1. In your QuPath project create rectangle annotations, of "Training" and "Validation" classes.
2. Lock the rectangles (right click > Annotations > Lock). 
3. Draw your ground truth. You can also run cellpose with `createAnnotations()` in the builder to have a starting ground truth you can manually correct. 
4. The drawn ground truth annotations must have **no classes**.

After you have saved the project, you can run the Cellpose training template script in 
`Extensions > Cellpose > Cellpose training script template`

Or you can download [Cellpose_training_template.groovy](src/main/resources/scripts/Cellpose_training_template.groovy) from this repo and run it from the script editor.

### More training options
[All options in Cellpose](https://github.com/MouseLand/cellpose/blob/45f1a3c640efb8ca7d252712620af6f58d024c55/cellpose/__main__.py#L36) 
have not been transferred. 

In case that this might be of use to you, please [open an issue](https://github.com/BIOP/qupath-extension-cellpose/issues). 


### Training validation
You can find a [run-cellpose-qc.py](QC/run-cellpose-qc.py) python script in the `QC` folder of this repository. This is 
an adaptation of the Quality Control part of a [ZeroCostDL4Mic Notebook that was made for cellpose](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/Beta%20notebooks/Cellpose_2D_ZeroCostDL4Mic.ipynb).

Basically, when you train using this extension:
1. It will first train your model as expected
2. It will then run your newly trained cellpose model on your "Validation" images
3. At the end, it will run the [run-cellpose-qc.py](QC/run-cellpose-qc.py) python script to output validation metrics.
4. The validation metrics will be saved into a folder called `QC` in your QuPath Project


### Saving training results for publication purposes

In order to be as reproducible and sure of your results as possible, especially when it comes to publishing, these are 
our current guidelines:
1. Use `saveBuilder()` which saves a JSON file of your CellposeBuilder, which can be reused with `CellposeBuilder(File builderFile)`. That way you will not lose the setting your did
2. Save the `cellpose-training`, `QC` and `models` folders at the end of your training somewhere. This will contain everything that was made during training.
3. Save the training script as well.

## Prediction 

Running Cellpose is done via a script and is very similar to the excellent [QuPath StarDist Extension](https://github.com/qupath/qupath-extension-stardist)

You can find a template in QuPath in

`Extensions > Cellpose > Cellpose training script template`

Or you can download the [Cellpose_detection_template.groovy](src/main/resources/scripts/Cellpose_detection_template.groovy) script from this repo and place it in the script editor


All builder options that are implemented are [in the Javadoc](https://biop.github.io/qupath-extension-cellpose/)

### Breaking changes after QuPath 0.4.0
In order to make the extension more flexible and less dependent on the builder, a new Builder method `addParameter(name, value)` is available that can take [any cellpose CLI argument or argument pair](https://cellpose.readthedocs.io/en/latest/command.html#options). 
For this to work, some elements that were "hard coded" on the builder have been removed, so you will get some errors. For example: `excludeEdges()` and `clusterDBSCAN()` no longer exist. 
You can use `addParameter("exclude_on_edges")`, and `addParameter("cluster")` instead.

# Citing

Please cite this extension by linking it to this GitHub or to the release you used, and feel free to give us a star ⭐️

If you use this extension, you should cite the following publications

Stringer, C., Wang, T., Michaelos, M. et al **Cellpose: a generalist algorithm for cellular segmentation**. Nat Methods 18, 100–106 (2021). https://doi.org/10.1038/s41592-020-01018-x

Pachitariu, M., Stringer, C. **Cellpose 2.0: how to train your own model**. Nat Methods 19, 1634–1641 (2022). https://doi.org/10.1038/s41592-022-01663-4

Cutler, K.J., Stringer, C., Lo, T.W. et al. **Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation**. Nat Methods 19, 1438–1448 (2022). https://doi.org/10.1038/s41592-022-01639-4

Bankhead, P. et al. **QuPath: Open source software for digital pathology image analysis**. Scientific Reports (2017).
https://doi.org/10.1038/s41598-017-17204-5


# Building

You can build the QuPath Cellpose extension from source with

```bash
gradlew clean build
```

The output will be under `build/libs`.

* `clean` removes anything old
* `build` builds the QuPath extension as a *.jar* file and adds it to `libs`

# Notes and debugging

## Preprocessing your data, extracting Color Deconvolution stains

It has been useful to preprocess data to extract color-deconvolved channels feeding these to Cellpose, for example. This is where the `preprocess()` method is useful. 
Depending on the export, one might need to inform cellpose of which channel is to be considered nuclear and which channel cytoplasmic. The method `cellposeChannels()` helps to set the order, as in the example below.
```
def stains = getCurrentImageData().getColorDeconvolutionStains()
// ..
// .. builder is initialized before this line
.preprocess( ImageOps.Channels.deconvolve(stains),
             ImageOps.Channels.extract(0, 1) ) // 0 for HTX and 1 for DAB
. cellposeChannels(2, 1)                       // Use the second channel from the extracted image for the cytoplasm and the first channel for the nucleus in cellpose
```
## `Warn: No compatible writer found`
So far we experienced the `No compatible writer found` issue in the following cases:

1. The channel names were incorrect in the builder, so the image writer could not find the requested channels
2. The rectangles were too close or beyond the edges of the image, so there were no pixels to export
3. There were special characters in the file name, which caused it to fail.

## Overlap

In case you end up with split detections, this is caused by the overlap calculation not being done correctly or by setting the `.diameter()` to 0 in order for cellpose to determine it automatically.
In turn, this causes the QuPath extension to fail to extract tiles with sufficient overlap.
Use `setOverlap( int )` in the builder to set the overlap (in pixels) to a value 2x larger than the largest object you are segmenting.

### To find the overlap

You can draw a line ROI across your largest object in QuPath and run the following one-line script
```
print "Selected Line length is " + Math.round(getSelectedObject().getROI().getLength()) + " pixels"
```
Double whatever value is output from the script and use it in `setOverlap( int )` in the builder.

## Ubuntu Error 13: Permission Denied
[As per this post here](https://forum.image.sc/t/could-not-execute-system-command-in-qupath-thanks-to-groovy-script-and-java-processbuilder-class/61629/2?u=oburri), there is a permissions issue when using Ubuntu, which does not allow Java's `ProcessBuilder` to run. 
The current workaround is [to build QuPath from source](https://qupath.readthedocs.io/en/stable/docs/reference/building.html) in Ubuntu, which then allows the use of the `ProcessBuilder`, which is the magic piece of code that actually calls Cellpose.