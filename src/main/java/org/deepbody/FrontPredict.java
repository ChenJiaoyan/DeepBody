package org.deepbody;

import org.bytedeco.javacv.ImageTransformer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

/**
 * Created by john on 14.12.16.
 * Locate the position with the classified pixels
 */
public class FrontPredict {
    protected static String model_f = "Front_CNN.zip";
    protected static String predict_f = "290323911.jpg";

    public static void main(String args[]){
        File f = new File(System.getProperty("user.dir"),"src/main/resources/Body/Image/"+predict_f);
    }

    public static void predict(){

    }

    public static INDArray slide(){
        //dim: n * 3 * 64 * 64
        INDArray out  =null;

        return out;
    }
    public static void location(){

    }

    private static MultiLayerNetwork loadModel() throws IOException {
        File f = new File(System.getProperty("user.dir"),"src/main/resources/Body/"+model_f);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(f);
        return model;
    }
}
