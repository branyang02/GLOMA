from PIL import Image, ImageDraw, ImageFont

# Create a 512x512 black image
img = Image.new('RGB', (512, 512), 'black')

# Bounding boxes (format: [x1, y1, x2, y2])
bbox_motion = [166, 220, 203, 267]
bbox_reference = [268, 219, 297, 267]

suffix = "top"
img_num = 1

# llama
bbox_pred = [261, 171, 298, 218]
model = 'llama'

# ## chat
# bbox_pred = [268, 165, 297, 212]
# model = 'chat'

# Labels for bounding boxes
labels = ['motion', 'reference', 'pred']

# Draw the bounding boxes and labels
draw = ImageDraw.Draw(img)
draw.rectangle(bbox_motion, outline="red", width=2)
draw.rectangle(bbox_reference, outline="green", width=2)
draw.rectangle(bbox_pred, outline="blue", width=2)

# Add labels (approximately positioned near the top-left of each bounding box)
try:
    # Try to use a default font
    font = ImageFont.load_default()
except IOError:
    # If default font is not available, use a basic font with a fixed size
    font = ImageFont.truetype("arial", 15)

draw.text((bbox_motion[0], bbox_motion[1] - 20), labels[0], fill="red", font=font)
draw.text((bbox_reference[0], bbox_reference[1] - 20), labels[1], fill="green", font=font)
draw.text((bbox_pred[0], bbox_pred[1] - 20), labels[2], fill="blue", font=font)

# Display the image
output_path = f"/scratch/rhm4nj/GLOMA/GLOMA/gloma/{model}_{img_num}_{suffix}.jpg"  # Path to save the annotated image
img.save(output_path)
