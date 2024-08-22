import numpy as np
import torch


def extract_wavelet_coefficients(data_item, spatial_shapes, max_depth, keep_level, device = None):
    # data_item : (N, 1 + 7 * D, low, low, low)

    low = data_item[:, 0:1]
    highs = []
    for j in range(max_depth):
        spatial_shape = spatial_shapes[j+1]
        if j < keep_level:
            high = np.zeros((1, 1, 7, spatial_shape[0], spatial_shape[1], spatial_shape[2]))
        else:
            high = data_item[:,
                   (j - keep_level) * 7 + 1: (j - keep_level + 1) * 7 + 1]  # extract coefficients
            high = high[:, None]
        highs.append(high)

    low = torch.from_numpy(low).float()
    highs = [torch.from_numpy(high).float() for high in highs]

    if device is not None:
        low = low.to(device)
        highs = [high.to(device) for high in highs]

    return low, highs


def multi_dim_argsort(tensor, descending=True):
    tensor_size = list(tensor.size())
    tensor = tensor.reshape((-1))
    indices = torch.argsort(tensor, descending=descending)
    result_indices = []
    for size_i in tensor_size[::-1]:
        indices_i = indices % size_i
        indices = indices // size_i
        indices = indices.long()
        result_indices.append(indices_i.unsqueeze(1))

    result_indices = torch.cat(result_indices[::-1], dim=1)
    return result_indices


def create_coordinates(resolution, space_range, channel_dim = 7):
    channels_samples = np.linspace(0, channel_dim - 1, channel_dim)
    dimensions_samples = np.linspace(space_range[0], space_range[1], resolution)
    ch, x, y, z = np.meshgrid(channels_samples, dimensions_samples, dimensions_samples, dimensions_samples)
    ch, x, y, z = ch[:, :, :, :, np.newaxis], x[:, :, :, :, np.newaxis], y[:, :, :, :, np.newaxis], z[:, :, :, :, np.newaxis]
    coordinates = np.concatenate((ch, x, y, z), axis=4)
    coordinates = coordinates.reshape((-1, 4))
    coordinates = torch.from_numpy(coordinates).long()
    return coordinates

def extract_highs_from_colors(high_indices, high_values, max_depth, shape_list):
    cnt = 0
    highs_recon = []
    for idx in range(max_depth):
        order_expand = (max_depth - 1) - idx
        indices_len = 7 * ((2 ** order_expand) ** 3)
        high_values_idx = high_values[:, cnt:cnt + indices_len]
        high_indices_idx = high_indices[:, cnt:cnt + indices_len]

        padding_size = (shape_list[-1][0] * (2 ** order_expand) - shape_list[idx + 1][0]) // 2
        high_new = torch.zeros((1, 7, shape_list[idx + 1][0] + padding_size * 2,
                                shape_list[idx + 1][1] + padding_size * 2, shape_list[idx + 1][2] + padding_size * 2, 3)).to(high_indices_idx.device)

        ## reshape to fit the shape
        high_indices_idx = high_indices_idx.reshape((-1, 4)).long()
        high_values_idx = high_values_idx.reshape((-1, 3))

        # set the values
        high_new[0, high_indices_idx[:, 0], high_indices_idx[:, 1], high_indices_idx[:, 2], high_indices_idx[:,3], :] = high_values_idx

        high_new = high_new[:, :, padding_size:high_new.size(2) - padding_size,
                   padding_size:high_new.size(3) - padding_size,
                   padding_size:high_new.size(4) - padding_size, :]  # remove padding
        high_new = torch.permute(high_new, (0, 5, 1, 2, 3, 4))
        highs_recon.append(high_new)
        cnt += indices_len
    return highs_recon

def extract_highs_from_values(high_indices, high_values, max_depth, shape_list):
    cnt = 0
    highs_recon = []
    for idx in range(max_depth):
        order_expand = (max_depth - 1) - idx
        indices_len = 7 * ((2 ** order_expand) ** 3)
        high_values_idx = high_values[:, cnt:cnt + indices_len]
        high_indices_idx = high_indices[:, cnt:cnt + indices_len]

        padding_size = (shape_list[-1][0] * (2 ** order_expand) - shape_list[idx + 1][0]) // 2
        high_new = torch.zeros((1, 1, 7, shape_list[idx + 1][0] + padding_size * 2,
                                shape_list[idx + 1][1] + padding_size * 2, shape_list[idx + 1][2] + padding_size * 2)).to(high_indices_idx.device)

        ## reshape to fit the shape
        high_indices_idx = high_indices_idx.reshape((-1, 4)).long()
        high_values_idx = high_values_idx.reshape((-1))

        # set the values
        high_new[0, 0, high_indices_idx[:, 0], high_indices_idx[:, 1], high_indices_idx[:, 2], high_indices_idx[:,3]] = high_values_idx

        high_new = high_new[:, :, :, padding_size:high_new.size(3) - padding_size,
                   padding_size:high_new.size(4) - padding_size,
                   padding_size:high_new.size(5) - padding_size]  # remove padding
        highs_recon.append(high_new)
        cnt += indices_len
    return highs_recon


def pad_with_batch_idx(tensor):
    batch_size = tensor.size(0)
    padding_tensor = torch.arange(batch_size, device=tensor.device).long().unsqueeze(1).repeat(1, tensor.size(1)).unsqueeze(2)
    padded_tensor = torch.cat((padding_tensor, tensor), dim=-1)
    return padded_tensor

def extract_full_indices(device, max_depth, shape_list):
    highs_full_indices = []
    highs_full_indices_last = create_coordinates(shape_list[-1][0], space_range=(0, shape_list[-1][0] - 1),
                                                 channel_dim=1).to(device)
    for idx in range(max_depth):
        order_expand = (max_depth - 1) - idx
        highs_full_indices_idx = compute_level_indices(highs_full_indices_last, order_expand=order_expand)
        highs_full_indices_idx = highs_full_indices_idx.reshape((-1, 7 * ((2 ** order_expand) ** 3), 4))
        highs_full_indices.append(highs_full_indices_idx)
    highs_full_indices = torch.cat(highs_full_indices, dim=1)
    return highs_full_indices


def compute_level_indices(indices_keep_last, order_expand):
    indices_grid = create_coordinates(2 ** order_expand, (0, 2 ** order_expand - 1), channel_dim=7).to(
        indices_keep_last.device)
    mutiplier = torch.from_numpy(
        np.array([1, 2 ** order_expand, 2 ** order_expand, 2 ** order_expand])).unsqueeze(0).to(
        indices_keep_last.device)
    indices_keep = indices_grid.unsqueeze(0) + indices_keep_last.unsqueeze(1) * mutiplier
    return indices_keep