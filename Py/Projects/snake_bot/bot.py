import time
import argparse
from collections import deque

import numpy as np
import pyautogui


def select_point(prompt: str):
    print(prompt)
    print("把鼠标移动到目标位置并按回车（Enter）")
    input()
    return pyautogui.position()


def capture_region(bbox):
    img = pyautogui.screenshot(region=bbox)
    arr = np.array(img)
    # pyautogui returns RGB; convert to BGR for OpenCV conventions if needed
    return arr


def extract_board(img, grid_w, grid_h):
    h, w, _ = img.shape
    cell_w = w // grid_w
    cell_h = h // grid_h
    board = [[0 for _ in range(grid_w)] for __ in range(grid_h)]
    for r in range(grid_h):
        for c in range(grid_w):
            y1 = r * cell_h
            x1 = c * cell_w
            cell = img[y1:y1 + cell_h, x1:x1 + cell_w]
            mean = cell.mean(axis=(0, 1))  # RGB order
            r_mean, g_mean, b_mean = mean[0], mean[1], mean[2]
            # classify by color heuristics (may need tuning per game)
            if g_mean > r_mean + 30 and g_mean > 80:
                board[r][c] = 1  # snake
            elif r_mean > g_mean + 40 and r_mean > 100:
                board[r][c] = 2  # food
            else:
                board[r][c] = 0  # empty
    return board


def find_cells(board, val):
    res = []
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell == val:
                res.append((r, c))
    return res


def neighbors(pos, H, W):
    r, c = pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            yield (nr, nc)


def find_head(snake_cells, board):
    sset = set(snake_cells)
    # head tends to be an endpoint with only one adjacent snake cell
    for cell in snake_cells:
        cnt = 0
        for nb in neighbors(cell, len(board), len(board[0])):
            if nb in sset:
                cnt += 1
        if cnt == 1:
            return cell
    # fallback: return first
    return snake_cells[0] if snake_cells else None


def bfs_path(board, start, goal):
    H, W = len(board), len(board[0])
    q = deque([start])
    prev = {start: None}
    blocked = {tuple(p) for p in find_cells(board, 1)}
    while q:
        cur = q.popleft()
        if cur == goal:
            # reconstruct
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        for nb in neighbors(cur, H, W):
            if nb in prev:
                continue
            if nb in blocked and nb != goal:
                continue
            prev[nb] = cur
            q.append(nb)
    return None


def move_key(from_cell, to_cell):
    fr, fc = from_cell
    tr, tc = to_cell
    if tr < fr:
        return 'up'
    if tr > fr:
        return 'down'
    if tc < fc:
        return 'left'
    if tc > fc:
        return 'right'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-w', type=int, required=True, help='grid columns')
    parser.add_argument('--grid-h', type=int, required=True, help='grid rows')
    parser.add_argument('--sleep', type=float, default=0.08, help='loop sleep seconds')
    args = parser.parse_args()

    print('将提示你选择游戏区域（两个点：左上和右下）。')
    p1 = select_point('移动到左上角后按回车：')
    p2 = select_point('移动到右下角后按回车：')
    left = min(p1.x, p2.x)
    top = min(p1.y, p2.y)
    width = abs(p2.x - p1.x)
    height = abs(p2.y - p1.y)
    bbox = (left, top, width, height)

    print(f'已设定区域：{bbox}. 按 Ctrl+C 退出。')
    pyautogui.FAILSAFE = True

    try:
        while True:
            img = capture_region(bbox)
            board = extract_board(img, args.grid_w, args.grid_h)
            snake = find_cells(board, 1)
            food = find_cells(board, 2)
            if not snake or not food:
                time.sleep(args.sleep)
                continue
            head = find_head(snake, board)
            target = food[0]
            path = bfs_path(board, head, target)
            if path and len(path) >= 2:
                nk = move_key(path[0], path[1])
                if nk:
                    pyautogui.press(nk)
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print('退出')


if __name__ == '__main__':
    main()
