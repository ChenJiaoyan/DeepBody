package org.utils;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;

/**
 * Created by lisa on 1/2/17.
 */

public class DrawDotPanel {


    //ths method will colour a pixel red
    public static BufferedImage drawDot(BufferedImage img, ArrayList<int[]> locations) {
        for (int[] dot : locations) {
            int x = dot[0];
            int y = dot[1];
            if (x > 0 && y > 0 && y < img.getHeight() && x < img.getWidth()) {
                img.setRGB(x, y, 0xFFFF0000);
                if (x-1 > 0 ) {
                    img.setRGB(x-1, y, 0xFFFF0000);//red
                }
                if (y-1 > 0 ) {
                    img.setRGB(x, y-1, 0xFFFF0000);//red
                }
                if (x+1 < img.getWidth() ) {
                    img.setRGB(x+1, y, 0xFFFF0000);//red
                }
                if (y+1 < img.getHeight() ) {
                    img.setRGB(x, y+1, 0xFFFF0000);//red
                }
                System.out.printf("point: (%d, %d)\n",x,y);
            }
        }
        return img;
    }


    public static void showResult(File predict_f, ArrayList<int[]> locations) {
        int rate = 2;

        BufferedImage image = null;
        try {
            image = ImageIO.read(predict_f);
        } catch (IOException e) {
            e.printStackTrace();
        }
        BufferedImage img = drawDot(image, locations);
        Image dimg = img.getScaledInstance(img.getWidth() / rate, img.getHeight() / rate,
                Image.SCALE_SMOOTH);
        ImageIcon icon = new ImageIcon(dimg);
        JLabel imageLabel = new JLabel(icon);
        JFrame frame = new JFrame("Show Result");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(true);
        frame.getContentPane().add(imageLabel);
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String args[]) throws IOException {
        String location_file = "locations_front";
        File location_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + location_file);
        InputStreamReader reader = new InputStreamReader(new FileInputStream(location_f));
        BufferedReader br = new BufferedReader(reader);
        ArrayList<int[]> locations = new ArrayList<>();
        String line = br.readLine();
        String img_name = "207034429.jpg";
        File predict_f = new File(System.getProperty("user.dir"),
                "src/main/resources/Body/Prediction/Front/" + img_name);

        while (line != null) {
            if (line.startsWith(img_name)) {
                String[] tmp = line.split(";");
                for (int i = 1; i < tmp.length; i++) {
                    String[] tmp2 = tmp[i].split(",");
                    int[] loc = {Integer.parseInt(tmp2[0]), Integer.parseInt(tmp2[1])};
                    locations.add(loc);
                }
            }
            line = br.readLine();
        }
        showResult(predict_f, locations);
    }

}