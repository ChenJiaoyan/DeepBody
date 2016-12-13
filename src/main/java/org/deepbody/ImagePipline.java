package org.deepbody;


import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.BaseImageLoader;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;

/**
 * Created by john on 13.12.16.
 */
public class ImagePipline {

    protected static final long seed = 12345;
    //Images are of format given by allowedExtension -
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    public static final Random randNumGen = new Random(seed);

    public static void main(String args[]) throws FileNotFoundException {
        String filename = new ClassPathResource("/DataExamples/ImagePipeline/").getFile().getPath();
        File parentDir = new File(filename);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        System.out.println(filename);
    }
}
