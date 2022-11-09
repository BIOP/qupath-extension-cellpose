# QuPath Cellpose extension

This repo adds some support to use 2D Cellpose within QuPath through a Python virtual environment.


# Installation


## Step 1: Install Cellpose

Follow the instructions to install Cellpose from [the main Cellpose repository](https://github.com/mouseland/cellpose).
This extension will need to know the path to your Cellpose environment.


### NOTE: `scikit-image` Dependency
As of version 0.4 of this extension, QC (quality control) is run **automatically** when training a model.

Due to the dependencies of the validation code, located inside [run-cellpose-qc.py](QC/run-cellpose-qc.py) requires an 
extra dependency on `scikit-image`.

The simplest way to add it, is when installing Cellpose as instructed in the oficial repository and adding scikit-image
```bash
python -m pip install cellpose scikit-image
```


### Example Cellpose 2.0.5 installation with CUDA 11.3 GPU support

First, we create the conda environment:
```
conda create -n cellpose-205 python=3.8
conda activate cellpose-205
pip install cellpose==2.0.5 scikit-image==0.19.3
pip uninstall torch
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
```
<details>
  <summary>See the 'pip freeze' result</summary>

```
cellpose==2.0.5
certifi @ file:///C:/Windows/TEMP/abs_e9b7158a-aa56-4a5b-87b6-c00d295b01fanefpc8_o/croots/recipe/certifi_1655968940823/work/certifi
charset-normalizer==2.0.12
colorama==0.4.5
fastremap==1.13.0
idna==3.3
imagecodecs==2022.2.22
imageio==2.21.2
llvmlite==0.38.1
natsort==8.1.0
networkx==2.8.6
numba==0.55.2
numpy==1.22.4
opencv-python-headless==4.6.0.66
packaging==21.3
Pillow==9.1.1
pyparsing==3.0.9
PyWavelets==1.3.0
requests==2.28.0
scikit-image==0.19.3
scipy==1.8.1
tifffile==2022.5.4
torch==1.11.0+cu113
torchaudio==0.11.0+cu113
torchvision==0.12.0+cu113
tqdm==4.64.0
typing_extensions==4.2.0
urllib3==1.26.9
wincertstore==0.2
```
</details>
    
Next, we look for the Python executable, **which we will need later when configuring the QuPath Cellpose extension**.

```
where python
C:\Users\oburri\.conda\envs\cellpose-205\python.exe
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
> In case you do not do this step, Cellpose training will still work, but the QC step will be skipped and you will be notified that the `run-cellpose-qc.py` cannot be found.


## QuPath Extension Cellpose: First time setup

Go to `Edit > Preferences > Cellpose`
Complete the fields with the requested information. based on the `conda` installation above, this is what it should look like:
![Cellpose setup example](files/cellpose-qupath-setup-example.png)
> **Note**
> Prefer using "Python Executable" as Cellpose Environment Type, as this is more OS-agnostic than the other methods, which may become deprecated in the future.


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

After you have saved the project, you can run the Cellpose training in the following way:

```groovy
import qupath.ext.biop.cellpose.Cellpose2D

def cellpose = Cellpose2D.builder("cyto") // Can choose "None" if you want to train from scratch
                .channels("DAPI", "CY3")  // or use work with .cellposeChannels( channel1, channel2 ) and follow the cellpose way
//                .preprocess(ImageOps.Filters.gaussianBlur(1)) // Optional preprocessing QuPath Ops 
//                .epochs(500)             // Optional: will default to 500
//                .learningRate(0.2)       // Optional: Will default to 0.2
//                .batchSize(8)            // Optional: Will default to 8
//                .minTrainMasks(5)        // Optional: Will default to 5
                .useGPU()                 // Optional: Use the GPU if configured, defaults to CPU only
//                .modelDirectory( new File("My/folder/for/models")) // Optional place to store resulting model. Will default to QuPath project root, and make a 'models' folder
//                .saveBuilder("My Builder") // Optional: Will save a builder json file that can be reloaded with Cellpose2D.builder(File builderFile)
                .build()

// Once ready for training you can call the train() method
// train() will:
// 1. Go through the current project and save all "Training" and "Validation" regions into a temp folder (inside the current project)
// 2. Run the cellpose training via command line
// 3. Recover the model file after training, and copy it to where you defined in the builder, returning the reference to it
// 4. If it detects the run-cellpose-qc.py file in your QuPath Extensions Folder, it will run validation for this model

def resultModel = cellpose.train()

// Pick up results to see how the training was performed
println "Model Saved under "
println resultModel.getAbsolutePath().toString()

// You can get a ResultsTable of the training. 
def results = cellpose.getTrainingResults()
results.show("Training Results")

// You can get a results table with the QC results to visualize 
def qcResults = cellpose.getQCResults()
qcResults.show("QC Results")


// Finally you have access to a very simple graph 
cellpose.showTrainingGraph()
```


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

All builder options that are implemented are [in the Javadoc](https://biop.github.io/qupath-extension-cellpose/)

```groovy
import qupath.ext.biop.cellpose.Cellpose2D

// Specify the model name (cyto, nuc, cyto2, omni_bact or a path to your custom model)
def pathModel = 'cyto2'

def cellpose = Cellpose2D.builder( pathModel )
        .pixelSize( 0.5 )              // Resolution for detection
        .channels( 'DAPI' )            // Select detection channel(s)
//        .preprocess( ImageOps.Filters.median(1) )                // List of preprocessing ImageOps to run on the images before exporting them
//        .tileSize(2048)                // If your GPU can take it, make larger tiles to process fewer of them. Useful for Omnipose
//        .cellposeChannels(1,2)         // Overwrites the logic of this plugin with these two values. These will be sent directly to --chan and --chan2
//        .maskThreshold(-0.2)           // Threshold for the mask detection, defaults to 0.0
//        .flowThreshold(0.5)            // Threshold for the flows, defaults to 0.4 
//        .diameter(0)                   // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
//        .setOverlap(60)                // Overlap between tiles (in pixels) that the QuPath Cellpose Extension will extract. Defaults to 2x the diameter or 60 px if the diameter is set to 0 
//        .invert()                      // Have cellpose invert the image
//        .useOmnipose()                 // Add the --omni flag to use the omnipose segmentation model
//        .excludeEdges()                // Clears objects toutching the edge of the image (Not of the QuPath ROI)
//        .clusterDBSCAN()               // Use DBSCAN clustering to avoir over-segmenting long object
//        .cellExpansion(5.0)            // Approximate cells based upon nucleus expansion
//        .cellConstrainScale(1.5)       // Constrain cell expansion using nucleus size
//        .classify("My Detections")     // PathClass to give newly created objects
        .measureShape()                // Add shape measurements
        .measureIntensity()            // Add cell measurements (in all compartments)  
//        .createAnnotations()           // Make annotations instead of detections. This ignores cellExpansion
        .useGPU()                      // Optional: Use the GPU if configured, defaults to CPU only
        .build()

// Run detection for the selected objects
def imageData = getCurrentImageData()
def pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("Cellpose", "Please select a parent object!")
    return
}
cellpose.detectObjects(imageData, pathObjects)
println 'Done!'
```

# Citing

If you use this extension, you should cite the original Cellpose publication
- Stringer, C., Wang, T., Michaelos, M. et al. 
[*Cellpose: a generalist algorithm for cellular segmentation*](https://arxiv.org/abs/1806.03535)
Nat Methods 18, 100–106 (2021). https://doi.org/10.1038/s41592-020-01018-x

**You should also cite the QuPath publication, as described [here](https://qupath.readthedocs.io/en/stable/docs/intro/citing.html).**


# Building

You can build the QuPath Cellpose extension from source with

```bash
gradlew clean build
```

The output will be under `build/libs`.

* `clean` removes anything old
* `build` builds the QuPath extension as a *.jar* file and adds it to `libs`

# Notes and debugging

## Normalization

Like the StarDist extension, we provide a normalization option `normalizePercentiles`.
However, the QuPath Stardist Extension that keeps the normalized image in 32-bit with no clipping.
In turn, because Cellpose does its own normalization, there was no effect from using the normalization. 
However, the QuPath Cellpose Extension adds a clipping of values below 0.0 and above 1.0 to the normalization.

> Note however that QuPath normalizes channels jointly rather than independently, so you might get some odd results

 [Cellpose has implemented turning off their normalization, but it is not yet part of the current cellpose release](https://github.com/MouseLand/cellpose/issues).

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
