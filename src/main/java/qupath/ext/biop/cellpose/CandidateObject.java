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

import org.locationtech.jts.geom.Geometry;
import qupath.lib.objects.PathObject;
import qupath.lib.roi.GeometryTools;

/**
 * Holds a single candidate detection geometry, used to quickly check overlaps between detections.
 */
class CandidateObject {
    final double area;
    Geometry geometry;
    final PathObject parent; // Perhaps this duplicated things a bit, but we need it to sort the data

    CandidateObject(Geometry geom, PathObject parent) {
        this.geometry = geom;
        this.area = geom.getArea();
        this.parent = parent;

        // Clean up the geometry already
        geometry = GeometryTools.ensurePolygonal(geometry);

        // Keep only largest polygon?
        double maxArea = -1;
        int index = -1;

        for (int i = 0; i < geometry.getNumGeometries(); i++) {
            double area = geometry.getGeometryN(i).getArea();
            if (area > maxArea) {
                maxArea = area;
                index = i;
            }
        }
        geometry = geometry.getGeometryN(index);
    }
}
