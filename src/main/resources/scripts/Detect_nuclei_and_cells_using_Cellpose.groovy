package scripts
/* Last tested on QuPath-0.3.2
 * 
 * This scripts requires qupath-extension-cellpose 
 * cf https://github.com/BIOP/qupath-extension-cellpose
 */

// some qp that we need to detect objects and measure them
def imageData  = getCurrentImageData()
def server = getCurrentServer()
def cal = server.getPixelCalibration()
def downsample = 1.0

// if nothing Annotation is selected , let's create a full image annotation
def pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    createSelectAllObject(true)
}

clearDetections()

// Create a Cellpose detectors for cyto and nuclei
def pathModel_cyto = 'cyto2'
def cellpose_cyto = Cellpose2D.builder( pathModel_cyto )
        .channels("HCS","DAPI")
        .pixelSize( 0.3 )              // Resolution for detection
        .diameter(30)                  // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
        .measureShape()                // Add shape measurements
        .measureIntensity()            // Add cell measurements (in all compartments) 
        .useGPU()
        .build()

def pathModel_nuc = 'cyto2'
def cellpose_nuc = Cellpose2D.builder( pathModel_nuc )
        .channels("DAPI")
        .pixelSize( 0.3 )              // Resolution for detection
        .diameter(10)                  // Median object diameter. Set to 0.0 for the `bact_omni` model or for automatic computation
        .useGPU()
        .build()

// Run detection for the selected pathObjects and store resulting detections
cellpose_cyto.detectObjects(imageData, pathObjects)
cytos = getDetectionObjects()
cellpose_nuc.detectObjects(imageData, pathObjects)
nucs = getDetectionObjects()
//if one wants to check how each step is doing, uncomment the 4 lines below
//cytos.each{ it.setPathClass(getPathClass("Cyto"))}
//nucs.each{ it.setPathClass(getPathClass("Nuc"))}
//addObjects(cytos) // needed because cellpose detectors remove existing detections
//return

// make sure to clear everything 
clearDetections()

// Combine cytos and nuclei detections to create cell objects
// (we simply check that the nuclei center is inside the cell center) 
cells = []
cytos.each{ cyto ->
    nucs.each{ nuc ->      
        if ( cyto.getROI().contains( nuc.getROI().getCentroidX() , nuc.getROI().getCentroidY())){
            cells.add(PathObjects.createCellObject(cyto.getROI(), nuc.getROI(), getPathClass("Cellpose"), null ));
        }
    }
}
addObjects(cells)

// Intensity & Shape Measurements
// adapted from : https://forum.image.sc/t/transferring-segmentation-predictions-from-custom-masks-to-qupath/43408/12
def measurements = ObjectMeasurements.Measurements.values() as List
def compartments = ObjectMeasurements.Compartments.values() as List // Won't mean much if they aren't cells...
def shape = ObjectMeasurements.ShapeFeatures.values() as List
def cells = getCellObjects()
for ( cell in cells) {
    ObjectMeasurements.addIntensityMeasurements( server, cell, downsample, measurements, compartments )
    ObjectMeasurements.addCellShapeMeasurements( cell, cal,  shape )
}
fireHierarchyUpdate()
println 'Done!'

/*
 * imports
 */
import qupath.ext.biop.cellpose.Cellpose2D
import qupath.lib.analysis.features.ObjectMeasurements
