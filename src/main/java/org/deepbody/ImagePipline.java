package org.deepbody;


import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;

/**
 * Created by john on 13.12.16.
 */
public class ImagePipline {

    protected static final long seed = 12345;
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    public static final Random randNumGen = new Random(seed);

    protected static int height = 50;
    protected static int width = 50;
    protected static int channels = 3;
    protected static int outputNum = 4;

    public static void main(String args[]) throws IOException {
        String filename = new ClassPathResource("/DataExamples/ImagePipeline/").getFile().getPath();
        File parentDir = new File(filename);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        ImageTransform transform = new MultiImageTransform(randNumGen,new ShowImageTransform("Display - before "));
        recordReader.initialize(trainData,transform);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 2, 1, outputNum);

        while (dataIter.hasNext()) {
            DataSet ds = dataIter.next();
            System.out.println(ds);
            System.exit(0);
            try {
                Thread.sleep(3000);                 //1000 milliseconds is one second.
            } catch(InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
        }
        recordReader.reset();

        transform = new MultiImageTransform(randNumGen,new CropImageTransform(50), new ShowImageTransform("Display - after"));
        recordReader.initialize(trainData,transform);
        dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);
        while (dataIter.hasNext()) {
            DataSet ds = dataIter.next();
            System.out.println(ds);
            System.exit(0);
        }
        recordReader.reset();
    }
}
