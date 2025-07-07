plugins {
    id("com.gradleup.shadow") version "8.3.5"
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
}

// TODO: Configure your extension here (please change the defaults!)
qupathExtension {
    name = "qupath-extension-cellpose"
    group = "io.github.qupath"
    version = "0.11.0-SNAPSHOT"
    description = "QuPath extension to use Cellpose"
    automaticModule = "qupath.ext.cellpose"
}

dependencies {
    shadow(libs.qupath.gui.fx)
    shadow(libs.qupath.fxtras)
    shadow(libs.extensionmanager)
    shadow("commons-io:commons-io:2.15.0")
}

/*
 * Set HTML language and destination folder
 */
tasks.withType<Javadoc> {
    (options as StandardJavadocDocletOptions).addBooleanOption("html5", true)
    setDestinationDir(File(project.rootDir,"docs"))
}

/*
 * Avoid "Entry .gitkeep is a duplicate but no duplicate handling strategy has been set."
 * when using withSourcesJar()
 */
tasks.withType<org.gradle.jvm.tasks.Jar> {
    duplicatesStrategy = DuplicatesStrategy.INCLUDE
}