import cv2
import numpy as np


def multi_plot(height, width, rows, cols, images, titles,
               title_color=(255, 255, 255), center_title=False,
               background_color=(0, 0, 0),
               padding=10, centered=True, borders=False, border_color=(0, 255, 0)):
    """
    Plot the specified images with the specified titles in a matrix of images

    :type height: int
    :param height: Height of the output image
    :type width: int
    :param width: Width of the output image
    :type rows: int
    :param rows: Row count of the matrix
    :type cols: int
    :param cols: Column count of the matrix
    :type images: list[numpy.array]
    :param images: List of images to be displayed
    :type titles: list[string]
    :param titles: Titles corresponding to each image
    :raises: ValueError if number of titles is not equal with number of images
    """

    if len(images) != len(titles):
        raise ValueError("The number of titles is different from the number of images")
    
    text_size, baseline = cv2.getTextSize("ABCDEFGH", cv2.FONT_HERSHEY_PLAIN, 1, 1)
    title_height = text_size[1]
    disp_image = np.zeros((height, width, 3), np.uint8)
    disp_image[:, :] = background_color
    piece_height = int(float(height - (rows + 1) * (padding + title_height)) / rows)
    piece_width = int(float(width - (cols + 1) * padding) / cols)
    xpos, ypos = padding, padding
    for i in range(rows):
        ypos = padding
        for j in range(cols):
            if i * cols + j >= len(images):
                break
            curr_img = images[i * cols + j]
            curr_title = titles[i * cols + j]
            img_height, img_width = curr_img.shape[0], curr_img.shape[1]
            scale = min(float(piece_width) / img_width, float(piece_height) / img_height)
            new_height = int(img_height * scale)
            new_width = int(img_width * scale)

            h_diff = int(abs(piece_height - new_height) / 2) if centered else 0
            w_diff = int(abs(piece_width - new_width) / 2) if centered else 0
            # cvSetImageROI(disp_image, cvRect(...)));
            disp_image[
            xpos + h_diff + title_height:xpos + h_diff + title_height + new_height,
            ypos + w_diff:ypos + w_diff + new_width
            ] = cv2.resize(curr_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            text_size, baseline = cv2.getTextSize(curr_title, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            t_diff = int(abs(piece_width - text_size[0]) / 2) if center_title else 0
            # disp_image[xpos, ypos+w_diff:ypos+w_diff+new_width] = (255, 0, 0)
            cv2.putText(disp_image, curr_title, (ypos + w_diff + t_diff, xpos + title_height - 1),
                        cv2.FONT_HERSHEY_PLAIN, 1, title_color, 1)

            if borders:
                # Horizontal
                disp_image[xpos + title_height, ypos:ypos + piece_width] = border_color
                disp_image[((i + 1) * (piece_height + padding + title_height)) - 1,
                ypos:ypos + piece_width] = border_color
                # Vertical
                disp_image[xpos + title_height:xpos + title_height + piece_height, ypos] = border_color
                disp_image[xpos + title_height:xpos + title_height + piece_height,
                ((j + 1) * piece_width) - 1 + (j + 1) * padding] = border_color
            ypos += piece_width + padding
        xpos += piece_height + padding + title_height
    return disp_image