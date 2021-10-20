import qupath.ext.biop.cellpose.Cellpose2DTraining

// Example for training cellpose
// Requries a project, where there are annotations (usually rectangles) of class "Training" and "Validation" in which there are objects inside.
// The objects that you have annotated which will be exported as labeled images should have no PathClass at all.


def cellposeTrainer = new Cellpose2DTraining.Builder('cyto2')
        .channels("my", "channels") // Up to two channels for training.
        .pixelSize(1.2976)
        .epochs(500)
        .diameter(90) // Diameter for cellpose to further downsample your data to the desired model
        .modelDirectory("/where/my/model/should/be/saved")
        .build()


// train() will:
// 1. Go through the current project and save all "Training" and "Validation" regions into a temp folder (inside the current project)
// 2. Run the cellpose training via command line
// 3. Recover the model file after training, and copy it to where you defined in the builder, returning the reference to it

def resultModel = cellposeTrainer.train()

println "Model Saved under"+resultModel
