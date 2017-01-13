package org.deepbody;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
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
    private String normalizer_f;
    private String tiles_dir;
    private String ann_type;

    private long seed;
    private String[] allowedExtensions;
    private Random randNumGen;

    public static void main(String args[]) throws IOException {
        System.out.println("parameters: " + Arrays.toString(args));
        String ann_type = args[0];
        //String ann_type = "mln";
        int numEpochs = Integer.parseInt(args[1]);
        //int numEpochs = 10;
        int batchSize = Integer.parseInt(args[2]);
        //int batchSize = 45;
        FrontPixelClassification classification = new FrontPixelClassification(ann_type, numEpochs, batchSize);
        classification.learn();
    }

    public FrontPixelClassification(String ann_type, int numEpochs, int batchSize) throws IOException {


        this.ann_type = ann_type;
        //this.ann_type = "lenet";
        //this.numEpochs = 20;
        this.numEpochs = numEpochs;
        //this.batchSize = 40;
        this.batchSize = batchSize;

        iterations = 1;


        Properties properties = new Properties();
        InputStream inputStream = Thread.currentThread().getContextClassLoader().getResourceAsStream("config.properties");
        properties.load(inputStream);
        tile_height = Integer.parseInt(properties.getProperty("tile_height"));
        tile_width = Integer.parseInt(properties.getProperty("tile_width"));
        labelNum = Integer.parseInt(properties.getProperty("labelNum"));
        channels = Integer.parseInt(properties.getProperty("channels"));
        model_f = properties.getProperty("model_f");
        normalizer_f = properties.getProperty("normalizer_f");
        tiles_dir = properties.getProperty("tiles_dir");

        seed = 12345;
        allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        randNumGen = new Random(seed);

    }

    public void learn() throws IOException {
        System.out.println("**** Load Data ****");
        String filename = System.getProperty("user.dir") + "/src/main/resources/Body/" + tiles_dir;
        System.out.println(filename);
        File parentDir = new File(filename);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 70, 30);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        ImageRecordReader recordReader = new ImageRecordReader(tile_height, tile_width, channels, labelMaker);
        recordReader.initialize(trainData);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, labelNum);


        //DataNormalization normalizer = new ImagePreProcessingScaler(0, 1);
        //DataNormalization normalizer = new NormalizerMinMaxScaler();
        DataNormalization normalizer = new NormalizerStandardize();

        normalizer.fit(trainIter);
        trainIter.setPreProcessor(normalizer);

        System.out.println(trainIter.getLabels());

        System.out.println("**** Build Model ****");
        MultiLayerConfiguration conf = ANN_config();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.setListeners(new ScoreIterationListener(5));

        System.out.println("**** Train Model ****");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainIter);
        }

        System.out.println("**** Evaluate Model ****");
        recordReader.reset();
        recordReader.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, labelNum);
        testIter.setPreProcessor(normalizer);
        Evaluation eval = new Evaluation(labelNum);
        while (testIter.hasNext()) {
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }
        System.out.println(eval.stats());

        System.out.println("**** Save Model ****");
        storeModel(model);
        storeNormalizer(normalizer);
    }

    private MultiLayerConfiguration ANN_config() {
        MultiLayerConfiguration conf = null;
        double nonZeroBias;
        double dropOut;
        switch (ann_type) {
            case "alexnet":
                System.out.println("network type: alexnet");
                nonZeroBias = 1;
                dropOut = 0.5;
                conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0.0, 0.01))
                        .activation("relu").updater(Updater.NESTEROVS).iterations(1)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .learningRate(1e-2).biasLearningRate(1e-2 * 2)
                        .learningRateDecayPolicy(LearningRatePolicy.Step)
                        .lrPolicyDecayRate(0.1).lrPolicySteps(100000)
                        .regularization(true).l2(5 * 1e-4)
                        .momentum(0.9).miniBatch(false).list()
                        .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                        .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                        .layer(2, maxPool("maxpool1", new int[]{3, 3}))
                        .layer(3, conv5x5("cnn2", 256, new int[]{1, 1}, new int[]{2, 2}, nonZeroBias))
                        .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                        .layer(5, maxPool("maxpool2", new int[]{3, 3}))
                        .layer(6, conv3x3("cnn3", 384, 0))
                        .layer(7, conv3x3("cnn4", 384, nonZeroBias))
                        .layer(8, conv3x3("cnn5", 256, nonZeroBias))
                        .layer(9, maxPool("maxpool3", new int[]{3, 3}))
                        .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                        .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                        .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .name("output").nOut(labelNum).activation("softmax").build())
                        .setInputType(InputType.convolutionalFlat(tile_height, tile_width, channels))
                        .backprop(true).pretrain(false).build();
                break;
            case "lenet":
                System.out.println("network type: lenet");
                nonZeroBias = 1;
                dropOut = 0.5;
                conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .iterations(iterations)
                        .regularization(true).l2(5 * 1e-3)
                        .learningRate(.01)
                        .weightInit(WeightInit.XAVIER)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(Updater.NESTEROVS).momentum(0.9)
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                .nIn(channels)
                                .stride(1, 1)
                                .nOut(50)
                                .activation("identity")
                                .build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                        .layer(2, new ConvolutionLayer.Builder(5, 5)
                                .stride(1, 1)
                                .nOut(50)
                                .activation("identity")
                                .build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                        .layer(4, new DenseLayer.Builder().activation("relu")
                                .nOut(500).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(labelNum)
                                .activation("softmax")
                                .build())
                        .setInputType(InputType.convolutionalFlat(tile_height, tile_width, channels)) //See note below
                        .backprop(true).pretrain(false).build();
                break;
            default:
                System.out.println("network type: mln");
                conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .iterations(1)
                        .learningRate(0.006)
                        .updater(Updater.NESTEROVS).momentum(0.9)
                        .regularization(true).l2(1e-3)
                        .list()
                        .layer(0, new DenseLayer.Builder()
                                .nIn(tile_height * tile_width * 3).nOut(100).activation("relu")
                                .weightInit(WeightInit.XAVIER).build())
                        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(100).nOut(labelNum).activation("softmax")
                                .weightInit(WeightInit.XAVIER).build())
                        .pretrain(false).backprop(true)
                        .setInputType(InputType.convolutional(tile_height, tile_width, channels))
                        .build();
                break;
        }
        return conf;
    }

    private void storeModel(MultiLayerNetwork net) throws IOException {
        File f = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + model_f);
        boolean saveUpdater = true;
        ModelSerializer.writeModel(net, f, saveUpdater);
    }

    private void storeNormalizer(DataNormalization normalizer) throws IOException {
        File f0 = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + normalizer_f + "0");
        File f1 = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + normalizer_f + "1");
        File f2 = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + normalizer_f + "2");
        File f3 = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + normalizer_f + "3");
        normalizer.save(f0, f1, f2, f3);
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

    private static DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

}
