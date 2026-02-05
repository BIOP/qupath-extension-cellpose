plugins {
    id("maven-publish")
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
}

qupathExtension {
    name = "qupath-extension-cellpose"
    group = "io.github.qupath"
    version = "0.12.1-SNAPSHOT"
    description = "QuPath extension to use Cellpose"
    automaticModule = "qupath.ext.biop.cellpose"
}

dependencies {
    implementation(libs.qupath.gui.fx)
    implementation(libs.qupath.fxtras)
    implementation(libs.extensionmanager)
    implementation("commons-io:commons-io:2.15.0")
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

publishing {
    repositories {
        maven {
            name = "SciJava"
            val releasesRepoUrl = "https://maven.scijava.org/content/repositories/releases"
            val snapshotsRepoUrl = "https://maven.scijava.org/content/repositories/snapshots"
            url = if (project.hasProperty("release")) uri(releasesRepoUrl) else uri(snapshotsRepoUrl)
            credentials {
                username = System.getenv("MAVEN_USER")
                password = System.getenv("MAVEN_PASS")
            }
        }
    }

    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
            pom {
                licenses {
                    license {
                        name = "Apache License v2.0"
                        url = "http://www.apache.org/licenses/LICENSE-2.0"
                    }
                }
            }
        }
    }
}