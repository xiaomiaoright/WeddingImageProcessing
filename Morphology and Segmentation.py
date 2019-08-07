import numpy as np
from skimage import morphology as sk_mm
from matplotlib import pyplot as plt

# Required magic to display matplotlib plots in notebooks
%matplotlib inline


square = np.array([[0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]], dtype=np.uint8)


# Display it
fig = plt.figure(figsize=(3,3))
plt.imshow(square, cmap="binary")
plt.show()

#perform mathematical morphological operations on this by applying the following simple structuring element
struct_element = sk_mm.selem.diamond(1)
print(struct_element)

# Display it
fig = plt.figure(figsize=(1,1))
plt.imshow(struct_element, cmap="binary")
plt.show()


## Erosion
# Apply erosion->Erosion has the effect of removing pixels at the edges of shapes in the image. 
eroded_square = sk_mm.erosion(square, struct_element)
print(eroded_square)
fig = plt.figure(figsize=(6, 6))
# Plot original image
a=fig.add_subplot(1, 2, 1)
plt.imshow(square, cmap="binary")
a.set_title("Original")

# Plot eroded image
a=fig.add_subplot(1, 2, 2)
plt.imshow(eroded_square, cmap="binary")
a.set_title("Eroded")

plt.show()

## Dilation
#->Dilation has the effect of adding pixels at the edges of shapes in the image. 
# Apply erosion
#Apply dilation
dilated_square = sk_mm.dilation(square, struct_element)

# Display it
fig = plt.figure(figsize=(6, 6))

# Plot original image
a=fig.add_subplot(1, 2, 1)
plt.imshow(square, cmap="binary")
a.set_title("Original")

# Plot dilated image
a=fig.add_subplot(1, 2, 2)
plt.imshow(dilated_square, cmap="binary")
a.set_title("Dilated")

plt.show()

## Combine Dilation with Erosion
#-> Closed
dilated_square = sk_mm.dilation(square, struct_element)
closed_square = sk_mm.erosion(dilated_square, struct_element)

fig = plt.figure(figsize=(6, 6))

# Plot original image
a=fig.add_subplot(1, 2, 1)
image_plot_1 = plt.imshow(square, cmap="binary")
a.set_title("Original")

# Plot closed image
a=fig.add_subplot(1, 2, 2)
image_plot_2 = plt.imshow(closed_square, cmap="binary")
a.set_title("Closed")

plt.show()

#-> Opened
eroded_square = sk_mm.erosion(square, struct_element)
opened_square = sk_mm.dilation(eroded_square, struct_element)

# Display it
fig = plt.figure(figsize=(6, 6))

# Plot original image
a=fig.add_subplot(1, 2, 1)
image_plot_1 = plt.imshow(square, cmap="binary")
a.set_title("Original")

# Plot opened image
a=fig.add_subplot(1, 2, 2)
image_plot_2 = plt.imshow(opened_square, cmap="binary")
a.set_title("Opened")

plt.show()

# Opening and closing with Scikit-image
# Apply closing and opening
closed_square = sk_mm.closing(square, struct_element)
opened_square = sk_mm.opening(square, struct_element)

# Display it
fig = plt.figure(figsize=(9, 6))

# Plot original image
a=fig.add_subplot(1, 3, 1)
image_plot_1 = plt.imshow(square, cmap="binary")
a.set_title("Original")

# Plot closed image
a=fig.add_subplot(1, 3, 2)
image_plot_2 = plt.imshow(closed_square, cmap="binary")
a.set_title("Closed")

# Plot opened image
a=fig.add_subplot(1, 3, 3)
image_plot_2 = plt.imshow(opened_square, cmap="binary")
a.set_title("Opened")

plt.show()

