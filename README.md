# QuPath Cellpose extension

This repo adds some support to use 2D Cellpose within QuPath through a Python virtual environment.


## Installing

### Step 1: Install Cellpose

Follow the instructions to install Cellpose from [the main Cellpose repository](https://github.com/mouseland/cellpose).
This extension will need to know the path to your Cellpose environment.

#### Cellpose Environment on Mac
Currently, due to lack of testing and obscurity in the documentation, you cannot run the `conda` version of Cellpose from Mac or Linux, and a Python virtual environment is suggested (`venv`). Anyone able to run `conda activate` from a Java `ProcessBuilder`, please let me know :)

**The 'Simplest' way so far is the following**
1. Create a conda environment with the right Python version, and use that `conda` environment to create a `venv`
```
conda create python38 python=3.8
conda activate python38
python -m venv /where/you/want/your/cellpose
conda deactivate
source /where/you/want/your/cellpose/bin/activate
```
2. Now we are on the `venv` and we can install Cellpose
```
pip install cellpose 
```

### Step 1.1: Run Cellpose at least once from the command line

If you never ran Cellpose before, it needs to download its pretrained models the first time you run it. This may take some time and we've experienced the process hanging if done through QuPath. Just run cellpose from your command line and it should download all the models. Do this before using it in QuPath.

### Step 2: Install the QuPath Cellpose extension
Download the latest `qupath-extension-cellpose-[version].jar` file from [releases](https://github.com/biop/qupath-extension-cellpose/releases) and drag it onto the main QuPath window.

If you haven't installed any extensions before, you'll be prompted to select a QuPath user directory.
The extension will then be copied to a location inside that directory.

You might then need to restart QuPath (but not your computer).

### QuPath: First time setup
Go to `Edit > Preferences > Cellpose`
Complete the fields with the requested information
The example below is from a full GPU enabled cellpose installation that was made [by following the instructions here](https://c4science.ch/w/bioimaging_and_optics_platform_biop/computers-servers/software/gpu-deep-learning/python-venv/#cellpose). 
![image](https://user-images.githubusercontent.com/319932/137691866-2e15d4b5-526c-4360-9d1d-710bb285fd09.png)

In the example provided above for installing cellpose on Mac/Linux, you would enter `/where/you/want/your/cellpose/` and Python VENV as options

## Running

Running Cellpose is done via a script and is very similar to the excellent [QuPath StarDist Extension](https://github.com/qupath/qupath-extension-stardist)

```groovy
import qupath.ext.biop.cellpose.Cellpose2D

// Specify the model name (cyto, nuc, cyto2 or a path to your custom model)
def pathModel = 'cyto'

def cellpose = Cellpose2D.builder(pathModel)
        .probabilityThreshold(0.5)   // Probability (detection) threshold
        .channels('DAPI')            // Select detection channel(s)
        .normalizePercentiles(1, 99) // Percentile normalization
        .pixelSize(0.5)              // Resolution for detection
        .diameter(30)                // Average diameter of objects in px (at the requested pixel sie)
        .cellExpansion(5.0)          // Approximate cells based upon nucleus expansion
        .cellConstrainScale(1.5)     // Constrain cell expansion using nucleus size
        .measureShape()              // Add shape measurements
        .measureIntensity()          // Add cell measurements (in all compartments)
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

## Citing
If you use this extension, you should cite the original Cellpose publication
- Stringer, C., Wang, T., Michaelos, M. et al. 
[*Cellpose: a generalist algorithm for cellular segmentation*](https://arxiv.org/abs/1806.03535)
Nat Methods 18, 100–106 (2021). https://doi.org/10.1038/s41592-020-01018-x

**You should also cite the QuPath publication, as described [here](https://qupath.readthedocs.io/en/stable/docs/intro/citing.html).**


## Building
You can build the QuPath Cellpose extension from source with

```bash
gradlew clean build
```

The output will be under `build/libs`.

* `clean` removes anything old
* `build` builds the QuPath extension as a *.jar* file and adds it to `libs`

