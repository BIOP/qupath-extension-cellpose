/**
 * Cellpose Training Template script
 * @author Olivier Burri
 *
 * This script is a template to train a Cellpose model from QuPath.
 * It will:
 * 1. Go through the current project and save all "Training" and "Validation" regions into a temp folder (inside the current project)
 * 2. Run the cellpose training via command line with the parameters that you specify. See https://biop.github.io/qupath-extension-cellpose/qupath/ext/biop/cellpose/CellposeBuilder.html
 * for the parameters that you can specify through the extension and https://cellpose.readthedocs.io/en/latest/command.html for the Specific cellpose parameters
 * 3. Recover the model file after training, and copy it to where you defined in the builder, returning the location of the model file
 * 4. If it detects the run-cellpose-qc.py file in your QuPath Extensions Folder, it will run validation for this model
 * 5. It will return a ResultsTable with the training results and a graph of the training losses
 *
 * You can then use the model file to run Cellpose detection on your images
 *
 * NOTE that this template does not contain all options for training, But should help get you started
 */

// First we need to create a Cellpose2D builder and add all parameters that we want to use for training
def cellpose = Cellpose2D.builder("cyto") // Can choose "None" if you want to train from scratch
        .channels("DAPI", "CY3")  // or work with .cellposeChannels( channel1, channel2 ) and follow the cellpose way
//                .preprocess(ImageOps.Filters.gaussianBlur(1)) // Optional preprocessing QuPath Ops
//                .epochs(500)             // Optional: will default to 500
//                .groundTruthDirectory( new File("/my/ground/truth/folder")) // Optional: If you wish to save your GT elsewhere than the QuPath Project
//                .learningRate(0.2)       // Optional: Will default to 0.2
//                .batchSize(8)            // Optional: Will default to 8
//                .minTrainMasks(5)        // Optional: Will default to 5
//                .addParameter("save_flows")      // Any parameter from cellpose not available in the builder. See https://cellpose.readthedocs.io/en/latest/command.html
//                .addParameter("anisotropy", "3") // Any parameter from cellpose not available in the builder. See https://cellpose.readthedocs.io/en/latest/command.html
//                .modelDirectory( new File("My/folder/for/models")) // Optional place to store resulting model. Will default to QuPath project root, and make a 'models' folder
//                .saveBuilder("My Builder") // Optional: Will save a builder json file that can be reloaded with Cellpose2D.builder(File builderFile)
        .build()

// Now we can train a new model
def resultModel = cellpose.train()

// Pick up results to see how the training was performed
println "Model Saved under: "
println resultModel.getAbsolutePath().toString().replace('\\', '/') // To make it easier to copy paste in windows

// You can get a ResultsTable of the training.
def results = cellpose.getTrainingResults()
results.show("Training Results")

// You can get a results table with the QC results to visualize
def qcResults = cellpose.getQCResults()
qcResults.show("QC Results")

// Finally you have access to a very simple graph of the loss during training
cellpose.showTrainingGraph()

println "Training Script Finished"

import qupath.ext.biop.cellpose.Cellpose2D
