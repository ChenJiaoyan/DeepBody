package org.bodysample;


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
 * Created by john on 19.12.16.
 */
public class BodyTile {
    public static int height = 50;
    public static int width = 50;
    public static void main(String args[]){

    }

     public static void cut(String src_f,String des_f,int x,int y) throws IOException {
            FileInputStream is = null;
            ImageInputStream iis = null;
            try {
                is = new FileInputStream(src_f);
                Iterator<ImageReader> it = ImageIO.getImageReadersByFormatName("jpg");
                ImageReader reader = it.next();
                iis = ImageIO.createImageInputStream(is);
                reader.setInput(iis, true);
                ImageReadParam param = reader.getDefaultReadParam();
                Rectangle rect = new Rectangle(x, y, width, height);
                param.setSourceRegion(rect);
                BufferedImage bi = reader.read(0, param);
                ImageIO.write(bi, "jpg", new File(des_f));
            } finally {
                if (is != null)
                    is.close();
                if (iis != null)
                    iis.close();
            }
     }
}
