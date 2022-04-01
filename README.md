# QuPath Cellpose extension

This repo adds some support to use 2D Cellpose within QuPath through a Python virtual environment.


## Installing

### Step 1: Install Cellpose

Follow the instructions to install Cellpose from [the main Cellpose repository](https://github.com/mouseland/cellpose).
This extension will need to know the path to your Cellpose environment.

#### Cellpose Environment on Mac
Currently, due to lack of testing and obscurity in the documentation, you cannot run the `conda` version of Cellpose from Mac or Linux, and a Python virtual environment is suggested (`venv`). Anyone able to run `conda activate` from a Java `ProcessBuilder`, please let me know :)

**The 'Simplest' way with just a `venv` so far is the following**
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

### Step 1.1: Anaconda, Miniconda: Allow `conda` command from command line
1. Into the environment variable , edit PATH , add path to your `..\Anaconda3\condabin` default would be `C:\ProgramData\Anaconda3\condabin`
2. Open a new PowerShell (and/or PowerShell (x86) ) or Command Prompt and run the following command once to initialize conda: 
```
conda init
```

From now on you don't need to run a conda prompt you can simply activate a conda env from cmd.exe .

#### To check if it works:

1. Press the windows key, type cmd.exe (to get a command promt)
2. Type `conda env list` and you should get the list of your conda envs
 
### Step 1.2: Run Cellpose at least once from the command line

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

In the example provided above for installing cellpose on Mac/Linux, you would enter `/where/you/want/your/cellpose/` and `Python VENV` as options

## Using the Cellpose QuPath Extension

### Training

**Requirements**:
A QuPath project with rectangles of class "Training" and "Validation" inside which the ground truth objects have been painted.

We typically create a standalone QuPath project for training only. This project will contain the training images along with the ground truth annotations drawn in QuPath.
Here are some reasons we do it this way:
1. Separating training and prediction/analysis makes for clean project structures and easier sharing of the different steps of your workflow.
2. In case we need to get more ground truth, we can simply fire up the relevant QuPath project and import the newly trained model into any other project that might need it.

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
                .preprocess(ImageOps.Filters.gaussianBlur(1)) // Optional preprocessing QuPath Ops 
                .epochs(500)              // Optional: will default to 500
                .learningRate(0.2)        // Optional: Will default to 0.2
                .batchSize(8)             // Optional: Will default to 8
                .useGPU()                 // Optional: Use the GPU if configured, defaults to CPU only
                .modelDirectory(new File("My/location")) // Optional place to store resulting model. Will default to QuPath project root, and make a 'models' folder 
                .build()

// Once ready for training you can call the train() method
// train() will:
// 1. Go through the current project and save all "Training" and "Validation" regions into a temp folder (inside the current project)
// 2. Run the cellpose training via command line
// 3. Recover the model file after training, and copy it to where you defined in the builder, returning the reference to it

def resultModel = cellpose.train()

// Pick up results to see how the training was performed
println "Model Saved under "
println resultModel.getAbsolutePath().toString()

// You can get a ResultsTable of the training. 
def results = cellpose.getTrainingResults()
results.show("Training Results")

// Finally you have access to a very simple graph 
cellpose.showTrainingGraph()

```

**Extra training options:**
[All options in Cellpose](https://github.com/MouseLand/cellpose/blob/45f1a3c640efb8ca7d252712620af6f58d024c55/cellpose/__main__.py#L36) have not been transferred. 
In case that this might be of use to you, please [open an issue](https://github.com/BIOP/qupath-extension-cellpose/issues). 

## Prediction 

Running Cellpose is done via a script and is very similar to the excellent [QuPath StarDist Extension](https://github.com/qupath/qupath-extension-stardist)

All builder options are to be found [in the Javadoc](https://biop.github.io/qupath-extension-cellpose/)

```groovy
import qupath.ext.biop.cellpose.Cellpose2D

// Specify the model name (cyto, nuc, cyto2, omni_bact or a path to your custom model)
def pathModel = 'cyto'

def cellpose = Cellpose2D.builder( pathModel )
        .pixelSize( 0.5 )              // Resolution for detection
        .channels( 'DAPI' )            // Select detection channel(s)
//        .preprocess( ImageOps.Filters.median(1) )                // List of preprocessing ImageOps to run on the images before exporting them
//        .tileSize(2048)                // If your GPU can take it, make larger tiles to process fewer of them. Useful for Omnipose
//        .cellposeChannels(1,2)         // Overwrites the logic of this plugin with these two values. These will be sent directly to --chan and --chan2
//        .maskThreshold(-0.2)           // Threshold for the mask detection, defaults to 0.0
//        .flowThreshold(0.5)            // Threshold for the flows, defaults to 0.4 
//        .diameter(0)                   // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
//        .invert()                      // Have cellpose invert the image
//        .useOmnipose()                 // Add the --omni flag to use the omnipose segmentation model
//        .excludeEdges()                // Clears objects toutching the edge of the image (Not of the QuPath ROI)
//        .clusterDBSCAN()               // Use DBSCAN clustering to avoir over-segmenting long object
        .cellExpansion(5.0)            // Approximate cells based upon nucleus expansion
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

# Extra notes
Like the StarDist extension, we provide a normalization option `normalizePercentiles`.
However. because Cellpose does its own normalization, and QuPath keeps the normalized image in 32-bit with no clipping, there is no effect from using the normalization. 

In case you need your own normalization, you need to [ask Cellpose to implement it or allow to deactivate normalization](https://github.com/MouseLand/cellpose/issues).

# Ubuntu Error 13: Permission Denied
[As per this post here](https://forum.image.sc/t/could-not-execute-system-command-in-qupath-thanks-to-groovy-script-and-java-processbuilder-class/61629/2?u=oburri), there is a permissions issue when using Ubuntu, which does not allow Java's `ProcessBuilder` to run. The current workaround is [to build QuPath from source](https://qupath.readthedocs.io/en/stable/docs/reference/building.html) in Ubuntu, which then allows the use of the `ProcessBuilder`, which is the magic piece of code that actually calls Cellpose.
