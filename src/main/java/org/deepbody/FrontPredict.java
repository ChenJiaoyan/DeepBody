package org.deepbody;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
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
 * Created by john on 14.12.16.
 * Locate the position with the classified pixels
 */
public class FrontPredict {
    private static String model_f = "Front_CNN.zip";
    private static String predict_f = "290323911.jpg";

    private static final long seed = 12345;
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random randNumGen = new Random(seed);
    private static int tile_height = 64;
    private static int tile_width = 64;
    private static int slide_stride = 10;
    private static int img_height = 1280;
    private static int img_width = 960;
    private static int channels = 3;
    private static int outputNum = 3;
    private static int batchSize = 1;

    public static void main(String args[]) throws IOException {
        INDArray tmp = slide();
        System.out.println(Arrays.toString(tmp.shape()));
        System.exit(0);
        File f = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + model_f);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(f);
        model.output(tmp);
    }


    protected static INDArray slide() throws IOException {
        File img = new File(System.getProperty("user.dir"), "src/main/resources/Body/Image/" + predict_f);
        FileSplit filesInDir = new FileSplit(img, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit d = filesInDir.sample(pathFilter)[0];
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        ImageRecordReader recordReader = new ImageRecordReader(img_height, img_width, channels, labelMaker);
        recordReader.initialize(d);
        DataSetIterator it = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        scaler.fit(it);
        it.setPreProcessor(scaler);
        DataSet ds = it.next();

        int row_n = (int) Math.ceil((img_height-tile_height)/(double)slide_stride);
        int col_n = (int) Math.ceil((img_width-tile_width)/(double)slide_stride);
        INDArray m = ds.getFeatureMatrix().getRow(0);
        System.out.println(Arrays.toString(m.shape()));
        INDArray out = Nd4j.zeros(row_n,col_n,channels,tile_height,tile_width);
        for (int y = 0,r=0; y < img_height - tile_height; y = y + slide_stride,r=r+1) {
            for (int x = 0,c=0; x < img_width - tile_width; x = x + slide_stride,c=c+1) {
                INDArray tile = m.get(NDArrayIndex.all(), NDArrayIndex.interval(y, y + tile_height),
                        NDArrayIndex.interval(x, x + tile_width));
                out.get(NDArrayIndex.point(r),NDArrayIndex.point(c)).assign(tile);
            }
        }
        return out;
    }

}
