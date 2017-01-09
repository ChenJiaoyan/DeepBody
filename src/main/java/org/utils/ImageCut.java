package org.utils;

import javax.imageio.ImageIO;
import javax.imageio.ImageReadParam;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Iterator;

/**
 * Created by John on 1/9/17.
 */
public class ImageCut {
    public static void main(String args []) throws IOException {
        File src_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/Prediction/Front/207034429.jpg");
        File des_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/Prediction/Front/207034429_cut.jpg");
        FileInputStream is = null;
        ImageInputStream iis = null;
        int x1=308;
        int x2=762;
        int y1=98;
        int y2=1238;
        try {
            is = new FileInputStream(src_f);
            Iterator<ImageReader> it = ImageIO.getImageReadersByFormatName("jpg");
            ImageReader reader = it.next();
            iis = ImageIO.createImageInputStream(is);
            reader.setInput(iis, true);
            ImageReadParam param = reader.getDefaultReadParam();
            Rectangle rect = new Rectangle(x1, y1, x2-x1, y2-y1);
            param.setSourceRegion(rect);
            BufferedImage bi = reader.read(0, param);
            ImageIO.write(bi, "jpg", des_f);
        } finally {
            if (is != null)
                is.close();
            if (iis != null)
                iis.close();
        }
    }
}
