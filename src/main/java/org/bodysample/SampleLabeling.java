package org.bodysample;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.swing.*;

/**
 * Created by john on 20.12.16.
 */
public class SampleLabeling {
    protected static String label = "R_WAIST";
    protected static String img_f = "207034429.jpg"; //front: 207034429.jpg 290323911.jpg 1313883841.jpg
    protected static int width = 960;
    protected static int height = 1280;
    protected static int rate = 2;

    public static void main(String args[]) throws IOException {
        File f = new File(System.getProperty("user.dir"), "src/main/resources/Body/Image/front/" + img_f);
        BufferedImage image = ImageIO.read(f);

        //rescale the image to ensure the whole image is displayed
        //in sampling, do NOT enlarge the window
        Image dimg = image.getScaledInstance(width / rate, height / rate, Image.SCALE_SMOOTH);
        ImageIcon icon = new ImageIcon(dimg);
        JLabel imageLabel = new JLabel(icon);
        imageLabel.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                int x = e.getX() * rate;
                int y = e.getY() * rate;
                String s = img_f + "," + x + "," + y + "," + label;
                System.out.println(s);
                //JOptionPane.showMessageDialog(imageLabel, "x: " + x + ", y: " + y);
            }

            @Override
            public void mouseEntered(MouseEvent e) {
            }

            @Override
            public void mouseReleased(MouseEvent e) {
            }

            @Override
            public void mouseExited(MouseEvent e) {
            }

            @Override
            public void mousePressed(MouseEvent e) {
            }

        });
        imageLabel.validate();

        JFrame frame = new JFrame("Body Labeling Swing");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(imageLabel);
        frame.pack();
        frame.setVisible(true);

/*        int p = image.getRGB(0,0);
        int a = (p>>24) & 0xff;
        int r = (p>>16) & 0xff;
        int g = (p>>8) & 0xff;
        int b = p & 0xff;
        System.out.println(a + " " + r + " " + g + " " + b);
        File f2 = new File("/tmp/tmp.jpg");
        ImageIO.write(image,"jpg",f2);
        */
    }
}
