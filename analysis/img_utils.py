import svgwrite


def stack_images(images_path: list[str], width: float, height: float, filename: str):
    """Stack same size images vertically and export it"""

    nb_images = len(images_path)

    # Create a new SVG image
    result = svgwrite.Drawing(size=(height, width))

    # Add the first image
    for i, path in enumerate(images_path):
        img = result.add(svgwrite.image.Image(path))
        print(img)
        img.translate(tx=0, ty=i*height/nb_images)

    # Save the final image
    result.saveas(filename)

