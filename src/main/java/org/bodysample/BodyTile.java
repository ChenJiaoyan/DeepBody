package org.bodysample;

import javax.imageio.ImageIO;
import javax.imageio.ImageReadParam;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Iterator;

/**
 * Created by john on 19.12.16.
 */
public class BodyTile {
    protected static int height = 64;
    protected static int width = 64;
    protected static String type = "Front";
    protected static String sample_f = type + "_Sample_1";

    public static void main(String args[]) throws IOException {
        File f = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + sample_f);
        InputStreamReader reader = new InputStreamReader(new FileInputStream(f));
        BufferedReader br = new BufferedReader(reader);
        String line = br.readLine();
        while (line != null) {
            String[] tmp = line.split(",");
            String src_fname = tmp[0];
            int x = Integer.parseInt(tmp[1]);
            int y = Integer.parseInt(tmp[2]);
            String label = tmp[3];
            File src_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/Image/" + type+"/" +
                    src_fname);

            File des_dir = new File(System.getProperty("user.dir"), "src/main/resources/Body/Tiles_" + type +
                    "_1/" + label);
            if(!des_dir.exists()){
                des_dir.mkdir();
            }
            String des_fname = src_fname.substring(0, src_fname.length() - 4) + "_" + x + "_" + y + ".jpg";
            File des_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/Tiles_" + type + "_1/"
                    + label + "/" + des_fname);
            cut(src_f, des_f, x - width / 2, y - height / 2);
            line = br.readLine();
        }
    }

    public static void cut(File src_f, File des_f, int x, int y) throws IOException {
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
            ImageIO.write(bi, "jpg", des_f);
        } finally {
            if (is != null)
                is.close();
            if (iis != null)
                iis.close();
        }
    }
}
