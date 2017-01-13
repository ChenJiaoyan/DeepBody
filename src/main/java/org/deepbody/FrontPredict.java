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
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
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
    private File model_file;
    private File predict_file;
    private File normalizer_file0;
    private File normalizer_file1;
    private File normalizer_file2;
    private File normalizer_file3;

    private final String[] allowedExtensions;
    private Random randNumGen;
    private int tile_height;
    private int tile_width;
    private int img_height;
    private int img_width;
    private int channels;
    private int labelNum;
    private int otherLabel;

    private double decision_threshold;

    private int slide_stride;

    INDArray output;
    HashMap<Integer, int[]> locations;

    public static void main(String args[]) throws IOException {
        System.out.println("Parameters: " + Arrays.toString(args));
//        String img_file = "207034429.jpg";
        String img_file = args[0];
//        int slide_stride = 2;
        int slide_stride = Integer.parseInt(args[1]);
//        double decision_threshold = 0.8;
        double decision_threshold = Double.parseDouble(args[2]);

        System.out.println("#### Initialize ####");
        FrontPredict p = new FrontPredict(img_file, slide_stride, decision_threshold);
        System.out.println("#### Predict Tiles ####");
        p.predict_tile();
        System.out.println("#### Calculate Locations ####");
        p.cal_location();

        System.out.println("----------------------- Results ----------------------");
        HashMap<Integer, int[]> locations = p.getLocations();
        String result = img_file;
        for (int label = 0; label < p.getLabelNum(); label++) {
            if (label != p.getOtherLabel() && locations.containsKey(label)) {
                int[] location = locations.get(label);
                int r = location[0];
                int c = location[1];
                result = result + ";" + r + "," + c;
            }
        }
        System.out.println(result);
        System.out.println("----------------------- Results ----------------------");
    }

    public FrontPredict(String predict_file, int slide_stride, double decision_threshold) throws IOException {
        this.locations = new HashMap<>();

        this.slide_stride = slide_stride;
        this.decision_threshold = decision_threshold;
        this.predict_file = new File(System.getProperty("user.dir"),
                "src/main/resources/Body/Prediction/Front/" + predict_file);

        long seed = 12345;
        this.allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        this.randNumGen = new Random(seed);


        BufferedImage image = ImageIO.read(this.predict_file);
        this.img_height = image.getHeight();
        this.img_width = image.getWidth();

        Properties properties = new Properties();
        InputStream inputStream = Thread.currentThread().getContextClassLoader().getResourceAsStream("config.properties");
        properties.load(inputStream);
        this.tile_height = Integer.parseInt(properties.getProperty("tile_height"));
        this.tile_width = Integer.parseInt(properties.getProperty("tile_width"));
        this.labelNum = Integer.parseInt(properties.getProperty("labelNum"));
        this.channels = Integer.parseInt(properties.getProperty("channels"));
        this.otherLabel = Integer.parseInt(properties.getProperty("otherLabel"));

        String model_f = properties.getProperty("model_f");
        String normalizer_f = properties.getProperty("normalizer_f");
        this.model_file = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + model_f);
        this.normalizer_file0 = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + normalizer_f+"0");
        this.normalizer_file1 = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + normalizer_f+"1");
        this.normalizer_file2 = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + normalizer_f+"2");
        this.normalizer_file3 = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + normalizer_f+"3");

    }


    public void predict_tile() throws IOException {
        INDArray tiles = slide();
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(model_file);
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
        HashMap<Integer, ArrayList<int[]>> m = new HashMap<>();
        for (int r = 0; r < output.shape()[0]; r++) {
            for (int c = 0; c < output.shape()[1]; c++) {
                for (int label = 0; label < labelNum; label++) {
                    if (label == otherLabel) {
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
                                ArrayList<int[]> locs = m.get(label);
                                locs.add(loc);
                                m.put(label, locs);
                            }
                            break;
                        }

                    }
                }
            }
        }
        output_label_pixels(m);
        loc_decision_ANKLE_KNEE(m, true, "ANKLE");
        loc_decision_ANKLE_KNEE(m, false, "ANKLE");
        loc_decision_ANKLE_KNEE(m, true, "KNEE");
        loc_decision_ANKLE_KNEE(m, false, "KNEE");
        loc_decision_EYE_SHOULDER(m, true, "EYE");
        loc_decision_EYE_SHOULDER(m, false, "EYE");
        loc_decision_EYE_SHOULDER(m, true, "SHOULDER");
        loc_decision_EYE_SHOULDER(m, false, "SHOULDER");
    }

    private void loc_decision_average(HashMap<Integer, ArrayList<int[]>> m) {
        for (int label = 0; label < labelNum; label++) {
            if (label != otherLabel) {
                if (m.containsKey(label)) {
                    ArrayList<int[]> locs = m.get(label);
                    int x_sum = 0;
                    int y_sum = 0;
                    for (int i = 0; i < locs.size(); i++) {
                        x_sum += locs.get(i)[0];
                        y_sum += locs.get(i)[1];
                    }
                    int[] loc = {x_sum / locs.size(), y_sum / locs.size()};
                    this.locations.put(label, loc);
                } else {
                    int[] loc = {-1, -1};
                    this.locations.put(label, loc);
                }
            } else {
                int[] loc = {0, 0};
                this.locations.put(label, loc);
            }
        }
    }

    private void loc_decision_EYE_SHOULDER(HashMap<Integer, ArrayList<int[]>> m, boolean isLeft, String type) {
        int circle_r = 30;
        int label = 1;
        ArrayList<int[]> locs;
        if (type.equals("EYE")) {
            if (isLeft) {
                label = 1;
            } else {
                label = 6;
            }
            if (m.containsKey(label)) {
                locs = m.get(label);
                locs = height_filter(locs, img_height / 3, false);
            } else {
                return;
            }
        } else {
            if (isLeft) {
                label = 3;
            } else {
                label = 8;
            }
            if (m.containsKey(label)) {
                locs = m.get(label);
                locs = height_filter(locs, img_height / 2, false);
            } else {
                return;
            }
        }
        int max_num = 0;
        int[] max_loc = {-1, -1};
        for (int i = 0; i < locs.size(); i++) {
            int[] loc = locs.get(i);
            int num = count_surrounding_locs(locs, loc, circle_r);
            if (num > max_num) {
                max_num = num;
                max_loc = loc;
            }
        }
        int[] avg_loc = avg_surrounding_locs(locs, max_loc, circle_r);
        this.locations.put(label, avg_loc);
    }

    private void loc_decision_ANKLE_KNEE(HashMap<Integer, ArrayList<int[]>> m, boolean isLeft, String type) {
        int circle_r = 25;
        int label = 0;
        ArrayList<int[]> locs;
        if (type.equals("ANKLE")) {
            if (isLeft) {
                label = 0;
            } else {
                label = 5;
            }
            if (m.containsKey(label)) {
                locs = m.get(label);
                locs = height_filter(locs, 2 * img_height / 3, true);
            } else {
                return;
            }
        } else {
            if (isLeft) {
                label = 2;
            } else {
                label = 7;
            }
            if (m.containsKey(label)) {
                locs = m.get(label);
                locs = height_filter(locs, img_height / 2, true);
            } else {
                return;
            }
        }
        int max_num = 0;
        int[] max_loc = {-1, -1};
        for (int i = 0; i < locs.size(); i++) {
            int[] loc = locs.get(i);
            int num = count_surrounding_locs(locs, loc, circle_r);
            if (num > max_num) {
                max_num = num;
                max_loc = loc;
            }
        }
        int[] avg_loc = avg_surrounding_locs(locs, max_loc, circle_r);
        this.locations.put(label, avg_loc);
    }

    private int count_surrounding_locs(ArrayList<int[]> locs, int[] loc, int a) {
        int num = 1;
        int x0 = loc[0];
        int y0 = loc[1];
        for (int i = 0; i < locs.size(); i++) {
            int x = locs.get(i)[0];
            int y = locs.get(i)[1];
            if (x <= x0 + a && x >= x0 - a && y <= y0 + a && y >= y0 - a) {
                num++;
            }
        }
        return num;
    }

    private int[] avg_surrounding_locs(ArrayList<int[]> locs, int[] loc, int a) {
        int x0 = loc[0];
        int y0 = loc[1];
        int[] avg_loc = {x0, y0};
        int num = 1;
        for (int i = 0; i < locs.size(); i++) {
            int x = locs.get(i)[0];
            int y = locs.get(i)[1];
            if (x <= x0 + a && x >= x0 - a && y <= y0 + a && y >= y0 - a) {
                num++;
                avg_loc[0] += x;
                avg_loc[1] += y;
            }
        }
        avg_loc[0] = avg_loc[0] / num;
        avg_loc[1] = avg_loc[1] / num;
        return avg_loc;
    }

    private ArrayList<int[]> height_filter(ArrayList<int[]> locs, int threshold_line, boolean filter_upper) {
        ArrayList<int[]> new_locs = new ArrayList<>();
        for (int i = 0; i < locs.size(); i++) {
            int[] loc = locs.get(i);
            int y = loc[1];
            if (filter_upper && y > threshold_line) {
                new_locs.add(loc);
            }
            if (!filter_upper && y <= threshold_line) {
                new_locs.add(loc);
            }
        }
        return new_locs;
    }

    private void output_label_pixels(HashMap<Integer, ArrayList<int[]>> m) {
        Set<Integer> labels = m.keySet();
        Iterator<Integer> it = labels.iterator();
        while (it.hasNext()) {
            int label = it.next();
            ArrayList<int[]> locs = m.get(label);
            System.out.println("label: " + label);
            String result = predict_file.getPath();
            if (locs != null) {
                for (int i = 0; i < locs.size(); i++) {
                    int[] loc = locs.get(i);
                    result = result + ";" + loc[0] + "," + loc[1];
                }
            } else {
                result = result + ";None";
            }
            System.out.println(result);
        }
    }

    private INDArray slide() throws IOException {

        FileSplit filesInDir = new FileSplit(predict_file, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter);
        InputSplit is = filesInDirSplit[0];

        ImageRecordReader recordReader = new ImageRecordReader(img_height, img_width, channels, labelMaker);
        recordReader.initialize(is);
        DataSetIterator it = new RecordReaderDataSetIterator(recordReader, 1, 1, labelNum);

        //DataNormalization normalizer = new ImagePreProcessingScaler(0, 1);
        //DataNormalization normalizer = new NormalizerMinMaxScaler();
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.load(normalizer_file0,normalizer_file1,normalizer_file2,normalizer_file3);
        it.setPreProcessor(normalizer);
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


    public HashMap<Integer, int[]> getLocations() {
        return this.locations;
    }

    public int getLabelNum() {
        return this.labelNum;
    }

    public int getOtherLabel() {
        return this.otherLabel;
    }

}
