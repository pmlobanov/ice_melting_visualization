import pygame
import numpy as np
import random
from PIL import Image
import imageio

# --- Настройки ---
pygame.init()
width, height = 600, 400
cell_size = 2
cols, rows = width // cell_size, height // cell_size
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
frames = []

# --- Загрузка фонового изображения ---
background_img = pygame.image.load('background.png')
background_img = pygame.transform.scale(background_img, (width, height))

# --- Цвета ---
WATER_COLOR = (40, 40, 40)

# --- Инициализация сетки (float, 0.0 = сухо, 1.0 = вода) ---
grid = np.zeros((rows, cols), dtype=float)

# --- Начальная область воды по синей маске ---
img = Image.open('initial_mask.png').resize((cols, rows)).convert('RGB')
img_np = np.array(img)
blue = img_np[:, :, 2]
green = img_np[:, :, 1]
red = img_np[:, :, 0]
threshold = 200
water_mask = (red > threshold) & ( blue < 50 ) & (green < 50)
if water_mask.shape != (rows, cols):
    water_mask = water_mask.T
grid[water_mask] = 1.0

# --- Маска запретных зон ---
mask_img = Image.open('mask.png').convert('L').resize((cols, rows))
forbidden_mask = np.array(mask_img)
if forbidden_mask.shape != (rows, cols):
    forbidden_mask = forbidden_mask.T
forbidden_mask = forbidden_mask > 128

print("forbidden_mask shape:", forbidden_mask.shape)
print("grid shape:", grid.shape)

# --- Маска ограничения скорости --
# Загрузка маски замедлений (где красные области = замедление)
speed_img = Image.open('speed_mask.png').resize((cols, rows)).convert('RGB')
speed_np = np.array(speed_img)

# Выделяем красные области
red = speed_np[:, :, 0]
green = speed_np[:, :, 1]
blue = speed_np[:, :, 2]

# Условие: красный явно доминирует
red_mask = (red > threshold) & ( blue < 50 ) & (green < 50)
blue_mask = (blue > threshold) & (red < 50) & (green < 50)
threshold = 100  # подбери под свою задачу
green_mask = (green > threshold) & (green > red + 30) & (green > blue + 30)


# Формируем карту скоростей
speed_map = np.ones((rows, cols)) * 0.8  # базовая скорость
if red_mask.shape != (rows, cols):
    red_mask = red_mask.T
if blue_mask.shape != (rows, cols):
    blue_mask = blue_mask.T
if green_mask.shape != (rows, cols):
    green_mask = blue_mask.T
speed_map[blue_mask] = 2 # увеличение в скорости.
speed_map[green_mask] = 0.1 # небольшое замедление.
speed_map[red_mask] = 0.08  # замедление в красных областях

# --- Функция обновления сетки ---
def update_grid():
    new_grid = grid.copy()
    directions = [
        (0, 1, 0.20), (1, 0, 0.95), (-1, 0, 0.80), (0, -1, 0.50),
        (1, 1, 0.03), (-1, 1, 0.02), (1, -1, 0.50), (-1, -1, 0.01)
    ]
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if grid[y, x] > 0.7:
                for dx, dy, prob in directions:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < cols and 0 <= ny < rows and
                        not forbidden_mask[ny, nx] and
                        random.random() < prob * speed_map[ny, nx] and
                        grid[ny, nx] < 1.0):
                        # Добавляем шум: иногда не заливаем клетку
                        if random.random() < 0.4:
                            continue
                        new_grid[ny, nx] = min(1.0, new_grid[ny, nx] + random.uniform(0.2, 0.4))
    # Слабое размытие
    kernel = np.array([[0, 0.02, 0],
                       [0.02, 0.9, 0.02],
                       [0, 0.02, 0]])
    from scipy.signal import convolve2d
    blurred = convolve2d(new_grid, kernel, mode='same', boundary='wrap')
    return np.maximum(new_grid, blurred * 0.2)


# --- Основной цикл ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    grid = update_grid()

    # --- Отрисовка ---
    screen.blit(background_img, (0, 0))

    for y in range(rows):
        for x in range(cols):
            if grid[y, x] > 0.05:
                alpha = min(255, int(grid[y, x] * 230))
                color = (WATER_COLOR[0], WATER_COLOR[1], WATER_COLOR[2], alpha)
                surf = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                surf.fill(color)
                screen.blit(surf, (x * cell_size, y * cell_size))
            # elif forbidden_mask[y, x]:
            #     pygame.draw.rect(screen, (0, 200, 0), (x * cell_size, y * cell_size, cell_size, cell_size), 1)

    pygame.display.flip()
    clock.tick(10)

    # Сохраняем кадр
    frame = pygame.surfarray.array3d(screen)  # (width, height, 3)
    frame = np.transpose(frame, (1, 0, 2))  # (height, width, 3)
    frames.append(frame.copy())

pygame.quit()
# --- Сохраняем видео ---
imageio.mimsave('simulation.mp4', frames, fps=10)