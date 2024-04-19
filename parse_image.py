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
        
        return rgb_matrix

if __name__ == '__main__':
    image_path = 'blahblahblah.jpg' 
    matrix = image_to_rgb_matrix(image_path)
    print(matrix)
