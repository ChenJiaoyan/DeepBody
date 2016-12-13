package org.deepbody;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by john on 13.12.16.
 */
public class ImagePipline {
    public static void main(String args[]){
        INDArray arr1 = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});
        System.out.println("arr1:");
        System.out.println(arr1);
    }
}
