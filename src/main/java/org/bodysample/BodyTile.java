package org.bodysample;

import javax.imageio.ImageIO;
import javax.imageio.ImageReadParam;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Iterator;
import java.util.Properties;

/**
 * Created by john on 19.12.16.
 */
public class BodyTile {
    protected static int tile_height;
    protected static int tile_width;
    protected static String type = "Front";
    protected static String sample_f = type + "_Sample_1";

    public static void main(String args[]) throws IOException {
        Properties properties = new Properties();
        InputStream inputStream = Thread.currentThread().getContextClassLoader().getResourceAsStream("config.properties");
        properties.load(inputStream);
        tile_height = Integer.parseInt(properties.getProperty("tile_height"));
        tile_width = Integer.parseInt(properties.getProperty("tile_width"));

        File f = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + sample_f);
        InputStreamReader reader = new InputStreamReader(new FileInputStream(f));
        BufferedReader br = new BufferedReader(reader);
        String line = br.readLine();
        int num = 0;
        while (line != null) {
            String[] tmp = line.split(",");
            String src_fname = tmp[0];
            int x = Integer.parseInt(tmp[1]);
            int y = Integer.parseInt(tmp[2]);
            String label = tmp[3];

            File des_dir = new File(System.getProperty("user.dir"), "src/main/resources/Body/Tiles_" + type +
                    "_1/" + label);
            if (!des_dir.exists()) {
                des_dir.mkdir();
            }
            String des_fname = src_fname.substring(0, src_fname.length() - 4) + "_" + x + "_" + y + ".jpg";
            File des_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/Tiles_" + type + "_1/"
                    + label + "/" + des_fname);
            if (!des_f.exists()) {
                File src_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/Image/" + type + "/" +
                        src_fname);
                BufferedImage image = ImageIO.read(src_f);
                int img_width = image.getWidth();
                int img_height = image.getHeight();
                if (x - tile_width / 2 >= 0 && y - tile_height / 2 >= 0 &&
                        x + tile_width / 2 < img_width && y + tile_height / 2 < img_height) {
                    cut(src_f, des_f, x - tile_width / 2, y - tile_height / 2);
                    System.out.println(des_f);
                    num++;
                }
            }
            line = br.readLine();
        }
        System.out.println(num + " tiles added!");
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
            Rectangle rect = new Rectangle(x, y, tile_width, tile_height);
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
