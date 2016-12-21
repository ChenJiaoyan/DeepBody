package org.example;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by john on 21.12.16.
 */
public class ImageProcessing {
    protected static final long seed = 12345;
    protected static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    public static final Random randNumGen = new Random(seed);
    protected static int height = 50;
    protected static int width = 50;
    protected static int channels = 3;
    protected static int outputNum = 3;
    protected static int batchSize = 1;

    public static void main(String args[]) throws IOException {
        String filename = new ClassPathResource("/DataExamples/ImagePipeline/labelA/image_0010.jpg").getFile().getPath();
        File img = new File(filename);

        FileSplit filesInDir = new FileSplit(img, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter);
        InputSplit d = filesInDirSplit[0];

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(d);
        DataSetIterator it = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        scaler.fit(it);
        it.setPreProcessor(scaler);

        DataSet ds = it.next();
        System.out.println(Arrays.toString(ds.getFeatureMatrix().shape()));
        INDArray tmp = ds.getFeatureMatrix();
        tmp = tmp.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 2), NDArrayIndex.interval(0, 2, true));
        System.out.println(tmp);
        System.out.println(Arrays.toString(tmp.shape()));
        INDArray tmp2 = Nd4j.vstack(tmp, tmp);
        System.out.println(Arrays.toString(tmp2.shape()));

    }
}
