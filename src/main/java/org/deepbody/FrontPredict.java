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

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;

/**
 * Created by john on 14.12.16.
 */

public class FrontPredict {
    private File model_f;
    private File predict_f;

    private final String[] allowedExtensions;
    private Random randNumGen;
    private int tile_height;
    private int tile_width;
    private int img_height;
    private int img_width;
    private int channels;
    private int labelNum;

    private double decision_threshold;

    private int slide_stride;

    INDArray output;
    ArrayList<int[]> locations;

    public static void main(String args[]) throws IOException {
        System.out.println("Parameters: " + Arrays.toString(args));
//        String img_file = "207034429.jpg";
        String img_file = args[0];
//        int slide_stride = 2;
        int slide_stride = Integer.parseInt(args[1]);
//        double decision_threshold = 0.8;
        double decision_threshold = Double.parseDouble(args[2]);

        System.out.println("#### Initialize ####");
        FrontPredict p = new FrontPredict("Front_CNN_1.zip", img_file, slide_stride, decision_threshold);
        System.out.println("#### Predict Tiles ####");
        p.predict();
        System.out.println("#### Calculate Locations ####");
        p.cal_location();

        System.out.println("----------------------- Results ----------------------");
        ArrayList<int[]> locations = p.getLocations();
        String result = img_file;
        for (int i = 0; i < locations.size(); i++) {
            int[] location = locations.get(i);
            int r = location[0];
            int c = location[1];
            result = result + ";" + r + "," + c;
        }
        System.out.println(result);
        System.out.println("----------------------- Results ----------------------");
    }

    public FrontPredict(String model_file, String predict_file,
                        int slide_stride, double decision_threshold) throws IOException {
        this.model_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + model_file);
        this.predict_f = new File(System.getProperty("user.dir"),
                "src/main/resources/Body/Prediction/Front/" + predict_file);

        this.slide_stride = slide_stride;
        this.decision_threshold = decision_threshold;

        long seed = 12345;
        this.allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        this.randNumGen = new Random(seed);


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
    }

    //location of a body part: calculate the average row/column of all the pixels
    //that are predicted as that body part
    public void cal_location() {
        locations = new ArrayList<>();
        HashMap<Integer, ArrayList<int[]>> m = new HashMap<>();
        for (int r = 0; r < output.shape()[0]; r++) {
            for (int c = 0; c < output.shape()[1]; c++) {
                for (int label = 0; label < labelNum; label++) {
                    if (label == 5) {
                        continue;
                    } else {
                        if (output.getDouble(r, c, label) >= decision_threshold) {
                            int loc_y = r * slide_stride + tile_height / 2;
                            int loc_x = c * slide_stride + tile_width / 2;
                            int[] loc = {loc_x, loc_y};
                            if (!m.containsKey(label)) {
                                ArrayList<int[]> locs = new ArrayList<>();
                                locs.add(loc);
                                m.put(label, locs);
                            } else {
                                ArrayList<int []> locs = m.get(label);
                                locs.add(loc);
                                m.put(label,locs);
                            }
                            break;
                        }

                    }
                }
            }
        }
        average_locs(m);
    }

    private void average_locs(HashMap<Integer, ArrayList<int[]>> m) {
        for (int label = 0; label < labelNum; label++) {
            if (labelNum != 5) {
                if (m.containsKey(label)) {
                    ArrayList<int[]> locs = m.get(label);
                    int x_sum = 0;
                    int y_sum = 0;
                    for (int i = 0; i < locs.size(); i++) {
                        x_sum += locs.get(i)[0];
                        y_sum += locs.get(i)[1];
                    }
                    int[] loc = {x_sum / locs.size(), y_sum / locs.size()};
                    this.locations.add(loc);
                } else {
                    int[] loc = {-1, -1};
                    this.locations.add(loc);
                }
            } else {
                int[] loc = {0, 0};
                this.locations.add(loc);
            }
        }
    }

    private INDArray slide() throws IOException {

        FileSplit filesInDir = new FileSplit(predict_f, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter);
        InputSplit is = filesInDirSplit[0];

        ImageRecordReader recordReader = new ImageRecordReader(img_height, img_width, channels, labelMaker);
        recordReader.initialize(is);
        DataSetIterator it = new RecordReaderDataSetIterator(recordReader, 1, 1, labelNum);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(it);
        it.setPreProcessor(scaler);
        DataSet ds = it.next();

        int row_n = (int) Math.ceil((img_height - tile_height) / (double) slide_stride);
        int col_n = (int) Math.ceil((img_width - tile_width) / (double) slide_stride);
        INDArray m = ds.getFeatureMatrix().getRow(0);
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


    public ArrayList<int[]> getLocations() {
        return this.locations;
    }


}
