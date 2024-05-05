from PIL import Image, ImageDraw
import matplotlib.pyplot as plt 
import numpy as np

# x  = 768
# y = 1024
fname = 'coins'

with open(f"images/{fname}.txt",'r') as f:
    ln = (f.readline()[:-1]).split(" ")
    x = int(ln[0])
    y = int(ln[1])
    print(y, x)

def proc_file(img_path):
    print(img_path)
    img = np.zeros((y,x))
    with open(img_path, 'r') as f:
        for i in range(y):
            for j in range(x):
                inp = int(float(f.readline()[:-1]))
                # inp = max(inp, 0)
                # inp = 500 if (inp > 300) else 0
                if("radsym" in img_path):
                    # inp = 0 if (inp < 1000) else inp
                    inp = inp
                img[i][j] = inp + 1
    plt.imshow(img)
    plt.show()

def draw_circles(img_path):
    Ms = np.zeros((y,x))
    circles = np.zeros((y,x))
    img = Image.new('1', (y,x))
    draw = ImageDraw.Draw(img)

    with open(f"images_cuda/{fname}_gaussnms.txt", 'r') as f:
        for i in range(y):
            for j in range(x):
                inp = int(float(f.readline()[:-1]))
                Ms[i][j] = inp
    with open(f"images_cuda/{fname}_radii.txt", 'r') as f:
        for i in range(y):
            for j in range(x):
                rad = int(float(f.readline()[:-1]))
                if(Ms[i][j] > 2000):
                    draw.ellipse(
                        [(i-rad, j-rad), (i+rad, j+rad)], fill=200
                    )
    plt.imshow(np.transpose(img))
    plt.show()

# imp = f"images_cuda/{fname}_gray.txt"
# proc_file(imp)
# imp = f"images_cuda/{fname}_gradients.txt"
# proc_file(imp)
imp = f"images_cuda/{fname}_gaussnms.txt"
# proc_file(imp)
draw_circles("")
# imp = f"images_cuda/{fname}_radii.txt"
# proc_file(imp)