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
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Properties;
import java.util.Random;

/**
 * Created by john on 14.12.16.
 * Locate the position with the classified pixels
 */
public class FrontPredict {
    private File model_f;
    private File predict_f;

    private long seed;
    private final String[] allowedExtensions;
    private Random randNumGen;
    private int tile_height;
    private int tile_width;
    private int slide_stride;
    private int img_height;
    private int img_width;
    private int channels;
    private int labelNum;
    private int batchSize;

    INDArray output;
    ArrayList<int[]> locations;

    public static void main(String args[]) throws IOException {
        FrontPredict p = new FrontPredict("Body/Front_CNN_1.zip", "Body/Image/front/207034429.jpg");
        p.predict();
        p.showResult();
    }

    public FrontPredict(String model_file, String predict_file) throws IOException {
        this.model_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + model_file);
        this.predict_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/Prediction/" + predict_file);

        this.seed = 12345;
        this.allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        this.randNumGen = new Random(seed);

        this.batchSize = 114;

        BufferedImage image = ImageIO.read(this.predict_f);
        this.img_height = image.getHeight();
        this.img_width = image.getWidth();

        Properties properties = new Properties();
        InputStream inputStream = Thread.currentThread().getContextClassLoader().getResourceAsStream("config.properties");
        properties.load(inputStream);
        this.tile_height = Integer.parseInt(properties.getProperty("tile_height"));
        this.tile_width = Integer.parseInt(properties.getProperty("tile_width"));
        this.labelNum = Integer.parseInt(properties.getProperty("labelNum"));
        this.channels = Integer.parseInt(properties.getProperty("channels"));
        this.slide_stride = Integer.parseInt(properties.getProperty("slide_stride"));
    }


    public void predict() throws IOException {
        INDArray tiles = slide();
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(model_f);
        output = Nd4j.zeros(tiles.shape()[0], tiles.shape()[1], labelNum);
        for (int r = 0; r < tiles.shape()[0]; r++) {
            INDArray tiles_r = tiles.getRow(r);
            INDArray label_int = model.output(tiles_r);
            output.getRow(r).assign(label_int);
        }
        cal_location();
    }

    //location of a body part: calculate the average row/column of all the pixels
    //that are predicted as that body part
    private void cal_location() {
        locations = new ArrayList<>();
        int[] r_sums = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int[] c_sums = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int[] nums = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int r = 0; r < output.shape()[0]; r++) {
            for (int c = 0; c < output.shape()[1]; c++) {
                double maxL = 0.0;
                int labelL = 0;
                for (int label = 0; label < labelNum; label++) {
                    double l = output.getDouble(r, c, label);
                    if (l > maxL) {
                        maxL = l;
                        labelL = label;
                    }
                }

                // when the probability >= 0.8, we think it's a effective predict and take it for calculation
                if (maxL >= 0.8) {
                    nums[labelL]++;
                    r_sums[labelL] += r * slide_stride + 32; //r*slide_stride+32 is the estimate position on real image coordinate
                    c_sums[labelL] += c * slide_stride + 32;
                }
            }
        }
        for (int i = 0; i < labelNum; i++) {
            if (nums[i] != 0 && i != 5) {
                int[] loc = {r_sums[i] / nums[i], c_sums[i] / nums[i]};
                locations.add(loc);
            }
        }
    }


    private INDArray slide() throws IOException {
        FileSplit filesInDir = new FileSplit(predict_f, allowedExtensions, randNumGen);
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

        int row_n = (int) Math.ceil((img_height - tile_height) / (double) slide_stride);
        int col_n = (int) Math.ceil((img_width - tile_width) / (double) slide_stride);
        INDArray m = ds.getFeatureMatrix().getRow(0);
        System.out.println(Arrays.toString(m.shape()));
        INDArray out = Nd4j.zeros(row_n, col_n, channels, tile_height, tile_width);
        for (int y = 0, r = 0; y < img_height - tile_height; y = y + slide_stride, r = r + 1) {
            for (int x = 0, c = 0; x < img_width - tile_width; x = x + slide_stride, c = c + 1) {
                INDArray tile = m.get(NDArrayIndex.all(), NDArrayIndex.interval(y, y + tile_height),
                        NDArrayIndex.interval(x, x + tile_width));
                out.get(NDArrayIndex.point(r), NDArrayIndex.point(c)).assign(tile);
            }
        }
        return out;
    }

    /**
     * draw dots on predict image
     */
    public void showResult() {

        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(true);

        BufferedImage image = null;
        try {
            image = ImageIO.read(predict_f);
        } catch (IOException e) {
            e.printStackTrace();
        }
        DrawDotPanel panel = new DrawDotPanel(image);
        frame.add(panel);

        /**
         *  dot[1] is the x position, dot[0] is the y position
         */
        for (int[] dot : locations) {
            panel.drawDot(dot[1], dot[0]);
        }

        panel.repaint();
        frame.pack();
        frame.setVisible(true);
    }

}
