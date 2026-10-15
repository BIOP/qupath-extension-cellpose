/*-
 * Copyright 2020-2021 BioImaging & Optics Platform BIOP, Ecole Polytechnique Fédérale de Lausanne
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.biop.cellpose;

import org.apache.commons.io.FilenameUtils;
import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.RegionRequest;

import java.io.File;
import java.util.Collection;
import java.util.Collections;
import java.util.function.Supplier;

/**
 * Holds the correspondence between a {@link RegionRequest} and a saved input tile image on disk,
 * and infers the resulting Cellpose label ("mask") file name.
 * <p>
 * The public accessors are what the segmentation backends in
 * {@code qupath.ext.biop.cellpose.backend} need; candidate storage stays package-private.
 *
 * @author Olivier Burri
 */
public class TileFile {
    private final RegionRequest request;
    private final File imageFile;
    private final PathObject parent;

    /**
     * Produces the tile image on demand instead of reading it from {@link #imageFile}. Set only for
     * the in-process backend; null for the subprocess backend, which needs the file on disk.
     */
    private final Supplier<ImagePlus> imageSupplier;

    /**
     * Label image handed back in memory by the in-process backend. Null when the masks came from
     * (or must go to) {@link #getLabelFile()}.
     */
    private ImageProcessor labels;

    private Collection<CandidateObject> candidates = Collections.emptyList();

    TileFile(RegionRequest request, File imageFile, PathObject parent) {
        this(request, imageFile, parent, null);
    }

    TileFile(RegionRequest request, File imageFile, PathObject parent, Supplier<ImagePlus> imageSupplier) {
        this.request = request;
        this.parent = parent;
        this.imageFile = imageFile;
        this.imageSupplier = imageSupplier;
    }

    /**
     * @return the input tile image on disk, which the in-process backend never writes but still uses
     * to name the tile in logs. Use {@link #openImage()} to obtain the pixels.
     */
    public File getImageFile() {
        return imageFile;
    }

    /**
     * Obtain the tile image, either on demand or by reading the file Cellpose was given.
     *
     * @return the tile image, or null if it could not be produced
     */
    public ImagePlus openImage() {
        if (imageSupplier != null)
            return imageSupplier.get();
        return IJ.openImage(imageFile.getAbsolutePath());
    }

    /**
     * @return true if this tile is handled entirely in memory, with no tile or mask file on disk
     */
    public boolean isInMemory() {
        return imageSupplier != null;
    }

    /**
     * @return the label image produced for this tile, or null if it was written to
     * {@link #getLabelFile()} instead
     */
    public ImageProcessor getLabels() {
        return labels;
    }

    /**
     * Hand back the label image in memory, so no mask file is written or re-read.
     *
     * @param labels the label image
     */
    public void setLabels(ImageProcessor labels) {
        this.labels = labels;
    }

    /** Release the label pixels once candidates have been extracted. */
    void clearLabels() {
        this.labels = null;
    }

    /**
     * @return the Cellpose label file that is expected to be produced for this tile
     * (the input file name with the {@code _cp_masks.tif} suffix)
     */
    public File getLabelFile() {
        return new File(FilenameUtils.removeExtension(imageFile.getAbsolutePath()) + "_cp_masks.tif");
    }

    /**
     * @return the region request describing the location of this tile in the source image
     */
    public RegionRequest getTile() {
        return request;
    }

    /**
     * @return the real parent object of this tile
     */
    public PathObject getParent() {
        return parent;
    }

    Collection<CandidateObject> getCandidates() {
        return this.candidates;
    }

    void setCandidates(Collection<CandidateObject> candidates) {
        this.candidates = candidates;
    }
}
