package org.deepbody;


import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.Random;

/**
 * Created by john on 14.12.16.
 * Classify the image pixel into body parts: L_ANKLE, L_EYE, L_KNEE, L_SHOULDER, L_WAIST
 * R_ANKLE, R_EYE, R_KNEE, R_SHOULDER, R_WAIST, OTHER
 */
public class FrontPixelClassification {

    private int tile_height;
    private int tile_width;
    private int channels;
    private int labelNum;

    private int numEpochs;
    private int batchSize;
    private int iterations;
    private String model_f;
    private String tiles_dir;

    private long seed;
    private String[] allowedExtensions;
    private Random randNumGen;

    public static void main(String args[]) throws IOException {
        FrontPixelClassification f = new FrontPixelClassification();
        f.learn();
    }

    public FrontPixelClassification() throws IOException {
        numEpochs = 1;
        batchSize = 30;
        iterations = 1;
        model_f = "Front_CNN_1.zip";
        tiles_dir = "Tiles_Front_1";

        Properties properties = new Properties();
        InputStream inputStream = Thread.currentThread().getContextClassLoader().getResourceAsStream("config.properties");
        properties.load(inputStream);
        tile_height = Integer.parseInt(properties.getProperty("tile_height"));
        tile_width = Integer.parseInt(properties.getProperty("tile_width"));
        labelNum = Integer.parseInt(properties.getProperty("labelNum"));
        channels = Integer.parseInt(properties.getProperty("channels"));

        seed = 12345;
        allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        randNumGen = new Random(seed);

    }

    public void learn() throws IOException {

        System.out.println("**** Load Data ****");
        String filename = new ClassPathResource("/Body/"+tiles_dir).getFile().getPath();
        File parentDir = new File(filename);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 75, 25);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        ImageRecordReader recordReader = new ImageRecordReader(tile_height, tile_width, channels, labelMaker);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        recordReader.initialize(trainData);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, labelNum);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        System.out.println("**** Build Model ****");
        MultiLayerConfiguration conf = ANN_Config();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.setListeners(new ScoreIterationListener(10));

        System.out.println("**** Train Model ****");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainIter);
        }

        System.out.println("**** Evaluate Model ****");
        recordReader.reset();
        recordReader.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, labelNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);
        Evaluation eval = new Evaluation(labelNum);
        while (testIter.hasNext()) {
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }
        System.out.println(eval.stats());

        System.out.println("**** Save Model ****");
        storeModel(model);
    }

    private MultiLayerConfiguration ANN_Config() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .activation("relu")
                .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP).momentum(0.9)
                .list()
                .layer(0, convInit("cnn1", channels, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2, 2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2, 2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(labelNum)
                        .activation("softmax")
                        .build())
                .setInputType(InputType.convolutionalFlat(tile_height, tile_width, channels))
                .backprop(true).pretrain(false).build();
        return conf;
    }

    private void storeModel(MultiLayerNetwork net) throws IOException {
        File f = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + model_f);
        boolean saveUpdater = true;
        ModelSerializer.writeModel(net, f, saveUpdater);
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2, 2}).name(name).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5, 5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}).name(name).nOut(out).biasInit(bias).build();
    }

}
