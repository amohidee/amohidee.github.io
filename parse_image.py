from PIL import Image

def image_to_rgb_matrix(image_path):

    with Image.open(image_path) as img:
        img = img.convert('RGB')
        
        width, height = img.size
        
        rgb_matrix = []
        
        for y in range(height):
            row = []
            for x in range(width):
                rgb = img.getpixel((x, y))
                
                row.append(rgb)
            rgb_matrix.append(row)
        
        return rgb_matrix, width, height

if __name__ == '__main__':
    fname = 'cells'
    image_path = f'images/{fname}.jpg' 
    out_path = f'images/{fname}.txt'
    matrix, w, h = image_to_rgb_matrix(image_path)
    # print(matrix)
    with open(out_path, "w") as f:
        f.write(f"{w} {h}\n")
        for row in matrix:
            for el in row:
                s = f"{el[0]}\n{el[1]}\n{el[2]}\n"
                f.write(s)
    