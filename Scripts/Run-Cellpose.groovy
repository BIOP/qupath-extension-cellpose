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