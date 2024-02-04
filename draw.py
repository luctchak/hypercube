import cv2
import numpy as np
from card import Card
from constants import Constants

def get_masks(img: np.array) -> np.array:
   # Read the image    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the image to get a binary image
    _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_masks = []
    for contour in contours:
        # Exclude contour containing a corner :
        if min(contour[:, 0, 0]) == 0 or min(contour[:, 0, 1]) == 0 or max(contour[:, 0, 0]) == img.shape[1] - 1 or max(contour[:, 0, 1]) == img.shape[0] - 1:
            continue   
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        mask = cv2.bitwise_and(mask, img)
        all_masks.append(mask)
    # Remove from this mask the pixels that are not black in the original image
    return all_masks


def hatch_within_mask(img, mask, spacing=20, thickness=2, coef: int = 1):
    # Create a blank image with the same dimensions as the original image
    hatch_image = np.ones_like(img) * 255  # Fill with white background    
    if coef >= 0:
        coef += 1
    # Apply hatch pattern within the mask region
    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            if (y - coef*x) % (spacing*abs(2*coef)) <= 2*abs(coef)*thickness:  # Check if pixel is within the hatch pattern
                if mask[x, y][0] == 255:  # Check if pixel is within the mask
                    hatch_image[x, y] = Constants.GREY
    
    # Combine the original image and the hatched image using the mask
    return img & hatch_image

def bubbles_within_mask(img, mask,  spacing=20, thickness=2, coef: int = 1):
    bubble_img = np.ones_like(img) * 255
   # Apply hatch pattern within the mask region
    if coef >= 0:
        coef += 1 
    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            if (y - coef*x) % spacing*abs(coef) <= 2*abs(2*coef)*thickness and mask[x, y][0] == 255:
                if (coef*y + x)% spacing*abs(coef) <= 2*abs(coef)*thickness:
                    bubble_img[x, y] = Constants.GREY

    # Combine the original image and the bubble image using the mask
    return img & bubble_img

def fill_element(img: np.array, masks: list[np.array], fill: str) -> np.array:
    if fill == 'full':
        for mask in masks:
            # Set to grey for the mask
            img[mask > 0] = (mask * Constants.GREY)[mask > 0]
    elif fill == 'lines':
        for i, mask in enumerate(masks):
            img = hatch_within_mask(img, mask, coef=i - len(masks)//2)
    elif fill == 'bubbles':
        for i, mask in enumerate(masks):
            img = bubbles_within_mask(img, mask, coef=i - len(masks)//2)
    return img

def colorize(img: np.array, masks, color: tuple[int, int, int]) -> np.array:
    coefs = np.linspace(1, 0.6, len(masks))
    for index, mask in enumerate(masks):
        # Set to grey for the mask
        coef = coefs[index]
        new_color = list(color[i] * coef for i in range(3))
        edge_color = list(color[i] * 0.6 for i in range(3))
        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                if mask[x, y][0] >= 100 and abs(img[x, y][0] - 128) < 3:
                    img[x, y] = new_color
                    
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if sum(img[x, y]) == 0: 
                img[x, y] = edge_color
            
    return img

def make_final_card(img: np.array, quantity: int) -> np.array:
    w, h, _ = Constants.CARD_SIZE
    if quantity == 1:
        centers = [(w//2, h//2)]
    elif quantity == 2:
        img = cv2.resize(img, (2*img.shape[1]//3, 2*img.shape[0]//3))
        centers = [(w//3, h//2), (2*w//3, h//2)]
    elif quantity == 3:
        img = cv2.resize(img, (2*img.shape[1]//3, 2*img.shape[0]//3))
        centers = [(w//4, h//2), (2*w//4, h//2), (3*w//4, h//2)]
    elif quantity == 4:
        img = cv2.resize(img, (2*img.shape[1]//3, 2*img.shape[0]//3))
        centers = [(w//4, h//2), (w//2, h//4), (w//2, 3*h//4), (3*w//4, h//2)]
    final_card = np.ones(Constants.CARD_SIZE) * 255  # Fill with white background
    for center in centers:
        # Paste the symbol centered at center coordinates
        final_card[center[0]-img.shape[0]//2:center[0]+img.shape[0]//2, center[1]-img.shape[1]//2:center[1]+img.shape[1]//2] = img
    return final_card

def draw_card(card: Card, output_path: str) -> None:
    img = cv2.imread(f"./base_images/{card.symbol}.png")
    masks = get_masks(img)
    img = fill_element(img, masks, card.fill)
    img = colorize(img, masks, card.color)
    img = make_final_card(img, card.quantity)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = PIL.Image.fromarray(img)
    #img.save(output_path)
    cv2.imwrite(output_path, img)


if __name__ == '__main__':
    COLORS = [
        ("red", Constants.RED),
        ("green", Constants.GREEN),
        ("blue", Constants.BLUE),
        ("grey", Constants.GREY_STRONG)
    ]
    SYMBOLS = [
        "circle",
        "cube",
        "dedo",
        "tetra"
    ]
    FILLS = [
        "full",
        "lines",
        "bubbles",
        "empty"
    ]
    QUANTITIES = [1, 2, 3, 4]
    for symbol in SYMBOLS:
        for fill in FILLS:
            for name, color in COLORS:
                for quantity in QUANTITIES:
                    card = Card(symbol, quantity, color, fill)
                    draw_card(card, f'./output_images/{symbol}_{fill}_{name}_{quantity}.png')