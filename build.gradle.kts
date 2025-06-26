plugins {
    id("java-library")
    id("maven-publish")
    id("qupath.extension-conventions")
    id("qupath.javafx-conventions")
}

repositories {
    // Use this only for local development!
    mavenCentral()
    maven{
        url = uri("https://maven.scijava.org/content/repositories/releases")
    }
    maven{
        url = uri("https://maven.scijava.org/content/repositories/ome-releases")
    }
}


group = "io.github.qupath"
version = "0.10.2-SNAPSHOT"
description = "QuPath extension to use Cellpose"

var archiveBaseName = "qupath-extension-cellpose"
var moduleName ="qupath.extension.cellpose"

base {
    archivesName = archiveBaseName
    description = description
}

val qupathVersion = rootProject.version.toString()


dependencies {
    implementation(libs.qupath.fxtras)
    implementation("commons-io:commons-io:2.15.0")
    implementation(libs.bundles.logging)

    // For testing
    testImplementation(libs.junit)
}

/*dependencies {
    implementation "io.github.qupath:qupath-gui-fx:${qupathVersion}"
    implementation libs.qupath.fxtras
    implementation "commons-io:commons-io:2.15.0"
    implementation libs.bundles.logging
}*/

tasks.withType<ProcessResources> {
    from ("${projectDir}/LICENSE") {
        into("META-INF/licenses/")
    }
}

tasks.register<Sync>("copyResources") {
    description = "Copy dependencies into the build directory for use elsewhere"
    group = "QuPath"
    from(configurations.default)
    into("build/libs")
}


/*
 * Ensure Java 21 compatibility
 */
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
    if (project.properties["sources"] != null)
        withSourcesJar()
    if (project.properties["javadocs"] != null)
        withJavadocJar()
}


/*
 * Adding manifest information
 */
tasks {
    withType<Jar> {
        manifest {
            attributes["Implementation-Title"] = project.name
            attributes["Automatic-Module-Name"] = "${project.group}.$moduleName"
        }
    }
}

/*
 * Create javadocs for all modules/packages in one place.
 * Use -PstrictJavadoc=true to fail on error with doclint (which is rather strict).
 */
val strictJavadoc = findProperty("strictJavadoc")
if (strictJavadoc == false) {
    tasks.withType<Javadoc> {
        (options as StandardJavadocDocletOptions).addStringOption("Xdoclint:none", "-quiet")
    }
}

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

tasks.named<Test>("test") {
    useJUnitPlatform()
}

publishing {
    repositories {
        maven {
            name = "SciJava"
            val releasesRepoUrl = "https://maven.scijava.org/content/repositories/releases"
            val snapshotsRepoUrl = "https://maven.scijava.org/content/repositories/snapshots"
            url = uri(if(project.hasProperty("release")) releasesRepoUrl else snapshotsRepoUrl)
            credentials {
                username = System.getenv("MAVEN_USER")
                password = System.getenv("MAVEN_PASS")
            }
        }
    }

    publications {
        create<MavenPublication>("mavenJava") {
            groupId = "io.github.qupath"
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