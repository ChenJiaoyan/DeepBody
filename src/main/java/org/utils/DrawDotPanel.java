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

public class DrawDotPanel extends JPanel {

    private BufferedImage bimg;
    private Dimension dims;

    public DrawDotPanel(BufferedImage image) {
        bimg = image;
        dims = new Dimension(bimg.getWidth(), bimg.getHeight());
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.drawImage(bimg, 0, 0, null);
    }

    //this method will allow the changing of image
    public void setBufferedImage(BufferedImage newImg) {
        bimg = newImg;
    }

    //ths method will colour a pixel red
    public boolean drawDot(int x, int y) {

        if (x > dims.getHeight() || y > dims.getWidth()) {
            return false;
        }

        bimg.setRGB(x, y,  0xFFFF0000);//red

        repaint();
        return true;
    }

    @Override
    public Dimension getPreferredSize() {
        return dims;
    }

    public static void showResult(File predict_f, ArrayList<int []> locations) {

        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(true);

        BufferedImage image = null;
        try {
            image = ImageIO.read(predict_f);
        } catch (IOException e) {
            e.printStackTrace();
        }
        DrawDotPanel panel = new DrawDotPanel(image);
        frame.add(panel);

        /**
         *  dot[1] is the x position, dot[0] is the y position
         */
        for (int[] dot : locations) {
            panel.drawDot(dot[1], dot[0]);
        }

        panel.repaint();
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String args[]) throws IOException {
        String location_file = "locations_front";
        File location_f = new File(System.getProperty("user.dir"), "src/main/resources/Body/" + location_file);
        InputStreamReader reader = new InputStreamReader(new FileInputStream(location_f));
        BufferedReader br = new BufferedReader(reader);
        ArrayList<int []> locations = new ArrayList<>();
        String line = br.readLine();
        String img_name = "207034429.jpg";
        File predict_f = new File(System.getProperty("user.dir"),
                "src/main/resources/Body/Prediction/Front/" + img_name);

        while(line!=null){
            line = br.readLine();
            if(line.startsWith(img_name)){
                String [] tmp = line.split(";");
                for(int i=0;i<tmp.length;i++){
                    String [] tmp2 = tmp[i].split(",");
                    int [] loc = {Integer.parseInt(tmp2[0]),Integer.parseInt(tmp2[1])};
                    locations.add(loc);
                }
            }
        }
        showResult(predict_f,locations);
    }

}