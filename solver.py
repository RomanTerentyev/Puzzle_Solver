import os
import sys
import numpy as np

# dimensions of result image
W = 1200
H = 900
CHANNEL_NUM = 3  # we work with rgb images
MAX_VALUE = 255  # max pixel value, required by ppm header


def read_image(path):
    # second line of header contains image dimensions
    w, h = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int32)
    # skip 3 lines reserved for header and read image
    image = np.loadtxt(path, skiprows=3, dtype=np.int16).reshape((h, w, CHANNEL_NUM))
    return image


def write_image(path, img):
    h, w = img.shape[:2]
    # ppm format requires header in special format
    header = f'P3\n{w} {h}\n{MAX_VALUE}\n'
    with open(path, 'w') as f:
        f.write(header)
        for r, g, b in img.reshape((-1, CHANNEL_NUM)):
            f.write(f'{r} {g} {b} ')

            
def fitting(orig_tiles):
    tiles = orig_tiles.copy()
    # create a dictionary where for each side of the tile the best match from other tiles is determined
    fit_dict = {prim_index: {prim_side: -1 for prim_side in ('up', 'right', 'bottom', 'left')} for prim_index in range(len(tiles))}
    for prim_index in range(len(tiles)):    
        for prim_side in ('up', 'right', 'bottom', 'left'):   
            temp_dict = {}
            for sec_index in range(len(tiles)):
                if sec_index == prim_index:
                    continue
                for sec_side in ('bottom', 'left', 'up', 'right'):
                    side_dist = np.linalg.norm(tiles[prim_index][0] - tiles[sec_index][-1])
                    temp_dict[(sec_index, sec_side)] = side_dist
                    tiles[sec_index] = np.rot90(tiles[sec_index], k = 1)
            temp_index, temp_side = min(temp_dict, key=temp_dict.get)
            if fit_dict[temp_index][temp_side] != -1:
                if fit_dict[temp_index][temp_side]['value'] <= temp_dict[(temp_index, temp_side)]:
                    tiles[prim_index] = np.rot90(tiles[prim_index], k = 1)
                    continue
                else:
                    temp_dict_1 = fit_dict[temp_index][temp_side]
                    del temp_dict_1['value']
                    fit_dict[list(temp_dict_1.keys())[0]][list(temp_dict_1.values())[0]] = -1

                    temp_dict_1 = fit_dict[prim_index][prim_side]
                    if temp_dict_1 != -1:
                        del temp_dict_1['value']
                        fit_dict[list(temp_dict_1.keys())[0]][list(temp_dict_1.values())[0]] = -1
            fit_dict[prim_index][prim_side] = {temp_index: temp_side, 'value': temp_dict[(temp_index, temp_side)]}
            fit_dict[temp_index][temp_side] = {prim_index: prim_side, 'value': temp_dict[(temp_index, temp_side)]}
            tiles[prim_index] = np.rot90(tiles[prim_index], k = 1)
        
    # drop duplicates
    for prim_index in range(len(tiles)):
        temp_dict = fit_dict[prim_index]
        indexes = ('up', 'right', 'bottom', 'left')
        for i in range(len(indexes)-1):
            for j in range(i+1, len(indexes)):
                if temp_dict[indexes[i]] != -1 and temp_dict[indexes[j]] != -1:
                    if list(temp_dict[indexes[i]].keys())[0] == list(temp_dict[indexes[j]].keys())[0]:
                        if temp_dict[indexes[i]]['value'] > temp_dict[indexes[j]]['value']:
                            temp_dict[indexes[i]] = -1
                        else:
                            temp_dict[indexes[j]] = -1    
    return fit_dict


def roll_tile(tiles, fit_dict, index):
    # rotate the tile 90 degrees clockwise and change fit_dict accordingly
    tiles[index] = np.rot90(tiles[index], k = -1)
    a = fit_dict[index]
    a['up'], a['right'], a['bottom'], a['left'] = a['left'], a['up'], a['right'], a['bottom']
    for side, temp_dict in a.items():
        if temp_dict != -1:
            temp_index = list(temp_dict.keys())[0]
            temp_side = list(temp_dict.values())[0]
            try:
                fit_dict[temp_index][temp_side][index] = side
            except TypeError:
                pass

            
def turnovers(tiles, fit_dict):
    # find the tile which is best to start the assembly with
    min_mean_dist = 10000.0
    for prim_index, temp_dict in fit_dict.items():
        mean_dist = []
        for i in list(temp_dict.values()):
            if i != -1:
                mean_dist.append(i['value'])
        if len(mean_dist) > 2:
            if np.mean(mean_dist) < min_mean_dist:
                min_mean_dist = np.mean(mean_dist)
                first_tile = prim_index    
    
    # find the order of tiles from zero
    prev_stable_tiles = []
    stable_tiles = [first_tile]
    for _ in range(len(tiles)):
        temp_list = []
        for i in stable_tiles:
            if i in prev_stable_tiles:
                continue
            for val in list(fit_dict[i].values()):
                if val != -1:
                    if (list(val.keys())[0] not in prev_stable_tiles) and (list(val.keys())[0] not in temp_list):
                        temp_list.append(list(val.keys())[0])
        prev_stable_tiles = stable_tiles.copy()
        stable_tiles.extend(temp_list)
    stable_tiles = list(dict.fromkeys(stable_tiles))

    # rotate tiles to the correct state    
    rolled_tiles = set()
    for prim_index in stable_tiles:
        for prim_side, sec_side in zip(('up', 'right', 'bottom', 'left'), ('bottom', 'left', 'up', 'right')):
            if fit_dict[prim_index][prim_side] != -1:
                sec_index = list(fit_dict[prim_index][prim_side].keys())[0]
                if sec_index not in rolled_tiles:
                    for _ in range(4):
                        if list(fit_dict[prim_index][prim_side].values())[0] == sec_side:
                            break
                        roll_tile(tiles, fit_dict, sec_index)
                rolled_tiles.add(sec_index)
        rolled_tiles.add(prim_index)
    return stable_tiles


# find the correct order of tiles
def assembly(fit_dict, tiles, x_nodes, y_nodes, stable_tiles, W, H):
    size = max(len(x_nodes), len(y_nodes)) * 2 + 1
    order_list = np.full((size, size), -1)
    order_list[size//2, size//2] = stable_tiles[0]
    selected_tiles = stable_tiles.copy()
    selected_tiles.remove(stable_tiles[0])
    for index in stable_tiles:
        try:
            xi, yi = np.where(order_list == index)[0][0], np.where(order_list == index)[1][0]
        except IndexError:
            continue
        if order_list[xi-1, yi] == -1 and fit_dict[index]['up'] != -1:
            if list(fit_dict[index]['up'].keys())[0] in selected_tiles:
                order_list[xi-1, yi] = list(fit_dict[index]['up'].keys())[0]
                selected_tiles.remove(list(fit_dict[index]['up'].keys())[0])
                if len(selected_tiles) == 0: break
        if order_list[xi, yi+1] == -1 and fit_dict[index]['right'] != -1:
            if list(fit_dict[index]['right'].keys())[0] in selected_tiles:
                order_list[xi, yi+1] = list(fit_dict[index]['right'].keys())[0]
                selected_tiles.remove(list(fit_dict[index]['right'].keys())[0])
                if len(selected_tiles) == 0: break
        if order_list[xi+1, yi] == -1 and fit_dict[index]['bottom'] != -1:
            if list(fit_dict[index]['bottom'].keys())[0] in selected_tiles:
                order_list[xi+1, yi] = list(fit_dict[index]['bottom'].keys())[0]
                selected_tiles.remove(list(fit_dict[index]['bottom'].keys())[0])
                if len(selected_tiles) == 0: break
        if order_list[xi, yi-1] == -1 and fit_dict[index]['left'] != -1:
            if list(fit_dict[index]['left'].keys())[0] in selected_tiles:
                order_list[xi, yi-1] = list(fit_dict[index]['left'].keys())[0]
                selected_tiles.remove(list(fit_dict[index]['left'].keys())[0])
                if len(selected_tiles) == 0: break

    for i in range(2):
        temp_list = []
        for row in order_list:
            if np.mean(row) != -1.0:
                temp_list.append(row)
        if i == 0:
            order_list = np.rot90(np.array(temp_list), k = 1)
        else:
            order_list = np.rot90(np.array(temp_list), k = -1)
    if order_list.shape[0] > order_list.shape[1]:
        H, W = W, H    
    order_list = order_list.flatten()
    order_list = order_list[order_list != -1]
    if len(order_list) < len(stable_tiles):
        for i in stable_tiles:
            if i not in order_list:
                order_list = np.append(order_list, i)

    result_img = np.zeros((H, W, CHANNEL_NUM), dtype=np.uint8)
    dims = np.array([t.shape[:2] for t in tiles])
    h, w = np.min(dims, axis=0)
    x_nodes = np.arange(0, W, w)
    y_nodes = np.arange(0, H, h)
    xx, yy = np.meshgrid(x_nodes, y_nodes)
    nodes = np.vstack((xx.flatten(), yy.flatten())).T
    # fill grid with tiles
    for (x, y), index in zip(nodes, order_list):
        result_img[y: y + h, x: x + w] = tiles[index][:h, :w]    
    return result_img


def solve_puzzle(tiles_folder):        
    # read all tiles in list
    tiles = [read_image(os.path.join(tiles_folder, t)) for t in sorted(os.listdir(tiles_folder))]

    # scan dimensions of all tiles and find minimal height and width
    dims = np.array([t.shape[:2] for t in tiles])
    h, w = np.min(dims, axis=0)

    x_nodes = np.arange(0, W, w)
    y_nodes = np.arange(0, H, h)

    fit_dict = fitting(tiles)
    stable_tiles = turnovers(tiles, fit_dict)
    result_img = assembly(fit_dict, tiles, x_nodes, y_nodes, stable_tiles, W, H)
    if result_img.shape[0] > result_img.shape[1]:
        result_img = np.rot90(result_img, k = 1)

    output_path = "image.ppm"
    write_image(output_path, result_img)

    
if __name__ == "__main__":
    directory = sys.argv[1]
    solve_puzzle(directory)