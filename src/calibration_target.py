from reportlab.lib import colors
from reportlab.lib import pagesizes
from reportlab.lib import units
from reportlab.pdfgen import canvas
import sys


def make_target(filename, side=units.cm, margin=units.cm,
                pagesize=pagesizes.A4):
    target = canvas.Canvas(filename)
    height, width = pagesize
    num_rows = int((height - 2 * margin) // side)
    num_cols = int((width - 2 * margin) // side)
    for row in range(num_rows):
        for col in range(num_cols):
            x = row * side + margin
            y = col * side + margin
            if (row + col) % 2 == 0:
                fill = colors.white
            else:
                fill = colors.black
            target.setFillColorRGB(*fill.rgb())
            target.rect(x, y, side, side, fill=1)
    target.showPage()
    target.save()

if __name__ == '__main__':
    make_target(sys.argv[1])
