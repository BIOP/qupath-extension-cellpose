package scripts

import qupath.ext.biop.cellpose.Cellpose2D

/* Last tested on QuPath-0.3.2
 * 
 * This scripts requires the qupath-extension-cellpose 
 * https://github.com/BIOP/qupath-extension-cellpose
 * 
 * To use this script: 
 * 1. create "Training" and "Validation" rectangles in images in your project
 * 2. Create **annotations** inside these rectangles
 * 3. Save your images
 * 4. Run this script
 * 
 * It will create a folder 'cellpose-training' at the root of your QuPath folder
 * which you can then use for running cellpose training. 
 * Note that you can run cellpose training directly within QuPath as well
 * By using the script "Cellpose_training    .groovy"
 */
 
// Build a Cellpose instance for saving the image pairs
def cellpose = Cellpose2D.builder( "None" )     // No effect, as this script only exports the images
//                .channels( "DAPI", "CY3" )                        // Optional: Image channels to export
//                .preprocess( ImageOps.Filters.gaussianBlur( 1 ) ) // Optional: preprocessing QuPath Ops 
                .build()

// Just save the training images for cellpose, no training is made                
cellpose.saveTrainingImages()

println "\nTraining and validation images saved under:\n\n${cellpose.getTrainingDirectory()}\n${cellpose.getValidationDirectory()}\n"