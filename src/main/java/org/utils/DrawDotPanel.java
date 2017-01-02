package org.utils;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

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
}