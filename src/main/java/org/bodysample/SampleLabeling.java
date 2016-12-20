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
    protected static String label = "L_WAIST";
    protected static String img_f = "207034429.jpg";
    public static void main(String args []) throws IOException {
        File f = new File(System.getProperty("user.dir"),"src/main/resources/Body/Image/"+img_f);
        BufferedImage image = ImageIO.read(f);
        ImageIcon icon = new ImageIcon(image);
        JLabel imageLabel = new JLabel(icon);
        imageLabel.setSize(image.getWidth(null), image.getHeight(null));
        imageLabel.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                int x = e.getX();
                int y = e.getY();
                String s = img_f + "," + x + "," + y + "," + label;
                System.out.println(s);
                JOptionPane.showMessageDialog(imageLabel, "x: "+x+", y: "+y);
            }
            @Override
            public void mouseEntered(MouseEvent e){}
            @Override
            public void mouseReleased(MouseEvent e){}
            @Override
            public void mouseExited(MouseEvent e){}
            @Override
            public void mousePressed(MouseEvent e){}

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
