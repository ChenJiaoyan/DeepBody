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
import org.utils.DrawDotPanel;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by john on 14.12.16.
 * Locate the position with the classified pixels
 */
public class FrontPredict {
    private static String model_f = "Front_CNN.zip";
    private static String predict_f = "207034429.jpg";

    private static final long seed = 12345;
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random randNumGen = new Random(seed);
    private static int tile_height = 64;
    private static int tile_width = 64;
    private static int slide_stride = 10;
    private static int img_height = 1280;
    private static int img_width = 960;
    private static int channels = 3;
    private static int labelNum = 11;
    private static int batchSize = 114;

    public static void main(String args[]) throws IOException {
        INDArray tiles = slide();
        System.out.println(Arrays.toString(tiles.shape()));
        File f = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + model_f);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(f);
        INDArray output = Nd4j.zeros(tiles.shape()[0], tiles.shape()[1], labelNum);
        for(int r=0;r<tiles.shape()[0];r++){
            INDArray tiles_r = tiles.getRow(r);
            INDArray label_int = model.output(tiles_r);
            //double [] label_d = Doubles.toArray(Ints.asList(label_int));
            //INDArray output_r = Nd4j.create(label_d);
            output.getRow(r).assign(label_int);
        }
        ArrayList<int []> locations = location(output);
        /*System.out.println("L_ANKLE: " + Arrays.toString(locations.get(0)));
        System.out.println("L_EYE: " + Arrays.toString(locations.get(1)));
        System.out.println("L_KNEE: " + Arrays.toString(locations.get(2)));
        System.out.println("L_SHOULDER: " + Arrays.toString(locations.get(3)));
        System.out.println("L_WAIST: " + Arrays.toString(locations.get(4)));

        System.out.println("R_ANKLE: " + Arrays.toString(locations.get(6)));
        System.out.println("R_EYE: " + Arrays.toString(locations.get(7)));
        System.out.println("R_KNEE: " + Arrays.toString(locations.get(8)));
        System.out.println("R_SHOULDER: " + Arrays.toString(locations.get(9)));
        System.out.println("R_WAIST: " + Arrays.toString(locations.get(10)));*/
        showPredictResult(locations);
    }

    //location of a body part: calculate the average row/column of all the pixels
    //that are predicted as that body part
    private static ArrayList<int []> location(INDArray output){
        ArrayList<int []> locations= new ArrayList<>();
        int [] r_sums = {0,0,0,0,0,0,0,0,0,0,0};
        int [] c_sums = {0,0,0,0,0,0,0,0,0,0,0};
        int [] nums = {0,0,0,0,0,0,0,0,0,0,0};
        for(int r=0;r<output.shape()[0];r++){
            for(int c=0;c<output.shape()[1];c++){
                double maxL = 0.0;
                int labelL = 0;
                for(int label=0;label<labelNum;label++) {
                    double l = output.getDouble(r, c, label);
                    if(l > maxL) {
                        maxL = l;
                        labelL = label;
                    }
                }

                // when the probability >= 0.8, we think it's a effective predict and take it for calculation
                if(maxL >= 0.8) {
                    nums[labelL]++;
                    r_sums[labelL] += r * slide_stride + 32; //r*slide_stride+32 is the estimate position on real image coordinate
                    c_sums[labelL] += c * slide_stride + 32;
                }
            }
        }
        for(int i=0;i<labelNum;i++){
            if(nums[i] != 0 && i != 5) {
                int[] loc = {r_sums[i] / nums[i], c_sums[i] / nums[i]};
                locations.add(loc);
            }
        }
        return locations;
    }


    private static INDArray slide() throws IOException {
        File img = new File(System.getProperty("user.dir"), "src/main/resources/Body/Image/front/" + predict_f);
        BufferedImage image = ImageIO.read(img);
        img_height = image.getHeight();
        img_width = image.getWidth();
        FileSplit filesInDir = new FileSplit(img, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit d = filesInDir.sample(pathFilter)[0];
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        ImageRecordReader recordReader = new ImageRecordReader(img_height, img_width, channels, labelMaker);
        recordReader.initialize(d);
        DataSetIterator it = new RecordReaderDataSetIterator(recordReader, batchSize, 1, labelNum);
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

    /**
     * draw dots on predict image
     * @param predictResult
     */
    public static void showPredictResult(ArrayList<int[]> predictResult){

        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(true);

        File f = new File(System.getProperty("user.dir"), "src/main/resources/Body/Image/front/" + predict_f);
        BufferedImage image = null;
        try {
            image = ImageIO.read(f);
        } catch (IOException e) {
            e.printStackTrace();
        }
        DrawDotPanel panel = new DrawDotPanel(image);
        frame.add(panel);

        /**
         *  dot[1] is the x position, dot[0] is the y position
         */
        for(int[] dot:predictResult){
            panel.drawDot(dot[1], dot[0]);
        }

        panel.repaint();
        frame.pack();
        frame.setVisible(true);
    }

}
