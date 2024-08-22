import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from pytorch_wavelets.dwt import lowlevel as lowlevel
from torch.autograd import Function

import mcubes
import trimesh


class WaveletData(object):
    def __init__(
        self,
        shape_list,
        output_stage,
        max_depth,
        data_stage=None,
        low=None,
        highs_values=None,
        highs_indices=None,
        wavelet_volume=None,
    ):

        if data_stage is None:
            self.data_stage = max(1, output_stage)
        else:
            self.data_stage = data_stage

        self.max_depth = max_depth
        self.output_stage = output_stage
        self.shape_list = shape_list
        self.wavelet_volume = wavelet_volume
        self.highs_indices = (
            highs_indices[..., 1:] if highs_indices is not None else highs_indices
        )  # Extract only the last three dimension
        self.highs_values = highs_values
        self.low = low

    def convert_wavelet_volume(self):

        ### if wavelet volume is given then return it
        if self.wavelet_volume is not None:
            return self.wavelet_volume
        else:
            # if output stage == 0 --> C0 only
            if self.output_stage == 0:
                return self.low
            else:

                ## assertion
                assert self.low.size(0) == self.highs_values.size(
                    0
                ) and self.highs_values.size(0) == self.highs_indices.size(0)
                assert self.data_stage != 0  # if 0 -> the output stage should be 0
                assert self.highs_values.size(1) == self.highs_indices.size(1)

                batch_size = self.low.size(0)

                ### create zero array
                output_dim = 8**self.output_stage - 1  # compute
                data_dim = 8**self.data_stage - 1
                highs_volume = torch.zeros(
                    (
                        batch_size,
                        self.shape_list[-1][0],
                        self.shape_list[-1][1],
                        self.shape_list[-1][2],
                        data_dim,
                    )
                ).to(
                    self.low.device
                )  # output high volumes

                ### extend batch_size to the indices
                batch_pad = (
                    torch.arange(batch_size, device=self.highs_indices.device)
                    .unsqueeze(1)
                    .repeat((1, self.highs_indices.size(1)))
                    .unsqueeze(2)
                    .long()
                )  # B * P * 1
                high_indices_filled = torch.cat(
                    (batch_pad, self.highs_indices), dim=2
                )  # B * P * 4

                ### fill the values
                high_indices_filled = high_indices_filled.reshape((-1, 4))
                highs_volume[
                    high_indices_filled[:, 0],
                    high_indices_filled[:, 1],
                    high_indices_filled[:, 2],
                    high_indices_filled[:, 3],
                    :,
                ] = self.highs_values.reshape((-1, data_dim))

                ### remove those unwanted dim
                highs_volume = highs_volume[..., -output_dim:]

                ### change back to channel first and concat with low
                highs_volume = torch.permute(highs_volume, (0, 4, 1, 2, 3))
                wavelet_volume = torch.cat((self.low, highs_volume), dim=1)

                return wavelet_volume

    def convert_low_highs(self):

        ### compute the volume if not given
        if self.wavelet_volume is None:
            wavelet_volume = self.convert_wavelet_volume()
        else:
            wavelet_volume = self.wavelet_volume

        ## extract part
        low = wavelet_volume[:, :1]  ## extract low volume

        if self.output_stage == 0:  # fill with empty zeros
            batch_size = low.size(0)
            highs = batch_extract_highs_from_values(
                output_stage=self.output_stage,
                max_depth=self.max_depth,
                shape_list=self.shape_list,
                device=low.device,
                batch_size=batch_size,
            )
        else:
            highs_volume = wavelet_volume[:, 1:]

            output_dim = 8**self.output_stage - 1

            highs_indices = extract_full_indices(
                device=wavelet_volume.device,
                max_depth=self.max_depth,
                shape_list=self.shape_list,
            )  # N * 511 * 4 (N = 46^3)
            highs_indices = highs_indices[:, -output_dim:]  # N * O * 4 (N = 46^3)

            # padding
            batch_size = wavelet_volume.size(0)
            highs_indices = padding_high_indices(
                batch_size, highs_indices
            )  # B * N * O * 5

            ## convert high volume to channel list
            highs_volume = torch.permute(highs_volume, (0, 2, 3, 4, 1))
            highs_values = highs_volume.view(batch_size, -1, output_dim)

            ## obtain the highs
            highs = batch_extract_highs_from_values(
                output_stage=self.output_stage,
                max_depth=self.max_depth,
                shape_list=self.shape_list,
                device=wavelet_volume.device,
                highs_indices=highs_indices,
                highs_values=highs_values,
                batch_size=batch_size,
            )

        return low, highs

def prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row, device):
    g0_row, g1_row = lowlevel.prep_filt_sfb1d(g0_row, g1_row, device)
    g0_col, g1_col = lowlevel.prep_filt_sfb1d(g0_col, g1_col, device)
    g0_dep, g1_dep = lowlevel.prep_filt_sfb1d(g0_dep, g1_dep, device)

    g0_dep = g0_dep.reshape((1, 1, -1, 1, 1))
    g1_dep = g1_dep.reshape((1, 1, -1, 1, 1))
    g0_col = g0_col.reshape((1, 1, 1, -1, 1))
    g1_col = g1_col.reshape((1, 1, 1, -1, 1))
    g0_row = g0_row.reshape((1, 1, 1, 1, -1))
    g1_row = g1_row.reshape((1, 1, 1, 1, -1))

    return g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row


def prep_filt_afb3d(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row, device):
    h0_row, h1_row = lowlevel.prep_filt_afb1d(h0_row, h1_row, device)
    h0_col, h1_col = lowlevel.prep_filt_afb1d(h0_col, h1_col, device)
    h0_dep, h1_dep = lowlevel.prep_filt_afb1d(h0_dep, h1_dep, device)

    h0_dep = h0_dep.reshape((1, 1, -1, 1, 1))
    h1_dep = h1_dep.reshape((1, 1, -1, 1, 1))
    h0_col = h0_col.reshape((1, 1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, 1, -1))

    return h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row


def afb1d(x, h0, h1, mode="zero", dim=-1):
    """1D analysis filter bank (along one dimension only) of an image
    Inputs:
        x (tensor): 5D input with the last two dimensions the spatial input
        h0 (tensor): 5D input for the lowpass filter. Should have shape (1, 1,
            h, 1, 1) or (1, 1, 1, w, 1) or (1, 1, 1, 1, d)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w, 1) or (1, 1, 1, 1, d)
        mode (str): padding method can only be zero
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).
    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """

    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 5
    s = [1, 1, 1]
    s[d - 2] = 2
    s = tuple(s)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(
            np.copy(np.array(h0).ravel()[::-1]), dtype=torch.float, device=x.device
        )
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(
            np.copy(np.array(h1).ravel()[::-1]), dtype=torch.float, device=x.device
        )
    L = h0.numel()
    L2 = L // 2
    shape = [1, 1, 1, 1, 1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    assert mode in ["zero", "constant"]

    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode=mode)
    p = 2 * (outsize - 1) - N + L

    # Sadly, pytorch only allows for same padding before and after, if
    # we need to do more padding after for odd length signals, have to
    # prepad
    padding_mode = None
    if mode == "zero":
        padding_mode = "zero"
    elif mode == "constant":
        padding_mode = "replicate"
    else:
        raise Exception("Unknown mode")

    if p % 2 == 1:
        pad = [0, 0, 0, 0, 0, 0]
        pad[(4 - d) * 2 + 1] = 1
        pad = tuple(pad)
        if mode == "zero":
            function_padding = "constant"
            x = F.pad(x, pad, mode=function_padding, value=0.0)
        elif mode == "constant":
            function_padding = "replicate"
            x = F.pad(x, pad, mode=function_padding)
        else:
            raise Exception("Unknown mode")

    pad = [0, 0, 0]
    pad[d - 2] = p // 2
    pad = tuple(pad)
    # Calculate the high and lowpass
    if padding_mode == "zero":
        lohi = F.conv3d(x, h, padding=pad, stride=s, groups=C)
    else:
        pad_new = [pad[2 - i // 2] for i in range(6)]
        x = F.pad(x, pad_new, mode=padding_mode)
        lohi = F.conv3d(x, h, stride=s, groups=C)

    return lohi


def sfb1d(lo, hi, g0, g1, mode="zero", dim=-1):
    """1D synthesis filter bank of an image tensor"""
    C = lo.shape[1]
    d = dim % 5
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(
            np.copy(np.array(g0).ravel()), dtype=torch.float, device=lo.device
        )
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(
            np.copy(np.array(g1).ravel()), dtype=torch.float, device=lo.device
        )
    L = g0.numel()
    shape = [1, 1, 1, 1, 1]
    shape[d] = L
    N = 2 * lo.shape[d]

    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = [1, 1, 1]
    s[d - 2] = 2

    g0 = torch.cat([g0] * C, dim=0)
    g1 = torch.cat([g1] * C, dim=0)

    assert mode in ["zero", "constant"]

    pad = [0, 0, 0]
    pad[d - 2] = L - 2
    pad = tuple(pad)
    lo = lo.cuda()
    g0 = g0.cuda()
    hi = hi.cuda()
    g1 = g1.cuda()

    y = F.conv_transpose3d(
        lo, g0, stride=s, padding=pad, groups=C
    ) + F.conv_transpose3d(hi, g1, stride=s, padding=pad, groups=C)

    return y


class SFB3D(Function):
    """Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """

    @staticmethod
    def forward(ctx, low, highs, g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row, mode):
        mode = lowlevel.int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        hll, lhl, hhl, llh, hlh, lhh, hhh = torch.unbind(highs, dim=2)
        lll = low
        ## first level
        ll = sfb1d(lll, hll, g0_dep, g1_dep, mode=mode, dim=2)
        hl = sfb1d(lhl, hhl, g0_dep, g1_dep, mode=mode, dim=2)
        lh = sfb1d(llh, hlh, g0_dep, g1_dep, mode=mode, dim=2)
        hh = sfb1d(lhh, hhh, g0_dep, g1_dep, mode=mode, dim=2)

        ## second level
        l = sfb1d(ll, hl, g0_col, g1_col, mode=mode, dim=3)
        h = sfb1d(lh, hh, g0_col, g1_col, mode=mode, dim=3)

        ## last level
        y = sfb1d(l, h, g0_row, g1_row, mode=mode, dim=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row = ctx.saved_tensors
            dx = afb1d(dy, g0_row, g1_row, mode=mode, dim=4)
            dx = afb1d(dx, g0_col, g1_col, mode=mode, dim=3)
            dx = afb1d(dx, g0_dep, g1_dep, mode=mode, dim=2)
            s = dx.shape
            dx = dx.reshape(s[0], -1, 8, s[-3], s[-2], s[-1])
            dlow = dx[:, :, 0].contiguous()
            dhigh = dx[:, :, 1:].contiguous()
        return dlow, dhigh, None, None, None, None, None, None, None



class AFB3D(Function):
    """Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        h0_dep: depth lowpass
        h1_dep: depth highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W, D)
    """

    @staticmethod
    def forward(ctx, x, h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row, mode):
        ctx.save_for_backward(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row)
        ctx.shape = x.shape[-3:]
        mode = lowlevel.int_to_mode(mode)
        ctx.mode = mode
        lohi_dim_last = afb1d(x, h0_row, h1_row, mode=mode, dim=4)
        lohi_dim_last_2 = afb1d(lohi_dim_last, h0_col, h1_col, mode=mode, dim=3)
        y = afb1d(lohi_dim_last_2, h0_dep, h1_dep, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 8, s[-3], s[-2], s[-1])
        low = y[:, :, 0].contiguous()
        highs = y[:, :, 1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, lll, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_dep, h1_dep, h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            hll, lhl, hhl, llh, hlh, lhh, hhh = torch.unbind(highs, dim=2)

            ## first level
            ll = sfb1d(lll, hll, h0_dep, h1_dep, mode=mode, dim=2)
            hl = sfb1d(lhl, hhl, h0_dep, h1_dep, mode=mode, dim=2)
            lh = sfb1d(llh, hlh, h0_dep, h1_dep, mode=mode, dim=2)
            hh = sfb1d(lhh, hhh, h0_dep, h1_dep, mode=mode, dim=2)

            ## second level
            l = sfb1d(ll, hl, h0_col, h1_col, mode=mode, dim=3)
            h = sfb1d(lh, hh, h0_col, h1_col, mode=mode, dim=3)

            ## last level
            dx = sfb1d(l, h, h0_row, h1_row, mode=mode, dim=4)

            if dx.shape[-3] > ctx.shape[-3]:
                dx = dx[:, :, : ctx.shape[-3]]
            if dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:, :, :, : ctx.shape[-2]]
            if dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:, :, :, :, : ctx.shape[-1]]

        return dx, None, None, None, None, None


class DWTInverse3d(nn.Module):
    """Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """

    def __init__(self, J=1, wave="db1", mode="zero"):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            g0_dep, g1_dep = g0_col, g1_col

        # Prepare the filters
        filts = prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row, None)
        self.register_buffer("g0_dep", filts[0])
        self.register_buffer("g1_dep", filts[1])
        self.register_buffer("g0_col", filts[2])
        self.register_buffer("g1_col", filts[3])
        self.register_buffer("g0_row", filts[4])
        self.register_buffer("g1_row", filts[5])
        self.J = J
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(
                    ll.shape[0],
                    ll.shape[1],
                    7,
                    ll.shape[-3],
                    ll.shape[-2],
                    ll.shape[-1],
                    device=ll.device,
                )

            # 'Unpad' added dimensions
            if ll.shape[-3] > h.shape[-3]:
                ll = ll[..., :-1, :, :]
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            ll = SFB3D.apply(
                ll,
                h,
                self.g0_dep,
                self.g1_dep,
                self.g0_col,
                self.g1_col,
                self.g0_row,
                self.g1_row,
                mode,
            )
        return ll


class DWTForward3d(nn.Module):
    """Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """

    def __init__(self, J=1, wave="db1", mode="zero"):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
            h0_dep, h1_dep = h0_col, h1_col

        # Prepare the filters
        filts = prep_filt_afb3d(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row, None)
        self.register_buffer("h0_dep", filts[0])
        self.register_buffer("h1_dep", filts[1])
        self.register_buffer("h0_col", filts[2])
        self.register_buffer("h1_col", filts[3])
        self.register_buffer("h0_row", filts[4])
        self.register_buffer("h1_row", filts[5])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """Forward pass of the DWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = AFB3D.apply(
                ll,
                self.h0_dep,
                self.h1_dep,
                self.h0_col,
                self.h1_col,
                self.h0_row,
                self.h1_row,
                mode,
            )
            yh.append(high)

        return ll, yh


def batch_extract_highs_from_values(
    output_stage,
    max_depth,
    shape_list,
    device,
    batch_size,
    highs_indices=None,
    highs_values=None,
):
    cnt = 0
    highs_recon = []
    for idx in range(max_depth):
        current_stage = (max_depth - 1) - idx

        padding_size = (
            shape_list[-1][0] * (2**current_stage) - shape_list[idx + 1][0]
        ) // 2
        high_new = torch.zeros(
            (
                batch_size,
                1,
                7,
                shape_list[idx + 1][0] + padding_size * 2,
                shape_list[idx + 1][1] + padding_size * 2,
                shape_list[idx + 1][2] + padding_size * 2,
            )
        ).to(device)

        ### only those in output will be computed
        if current_stage < output_stage:
            assert highs_indices.size(0) == highs_values.size(0)

            ## reshape to fit the shape
            indices_len = 7 * ((2**current_stage) ** 3)
            high_values_idx = highs_values[:, :, cnt : cnt + indices_len]
            high_indices_idx = highs_indices[:, :, cnt : cnt + indices_len]
            high_indices_idx = high_indices_idx.reshape((-1, 5)).long()
            high_values_idx = high_values_idx.reshape((-1))

            # set the values
            high_new[
                high_indices_idx[:, 0],
                0,
                high_indices_idx[:, 1],
                high_indices_idx[:, 2],
                high_indices_idx[:, 3],
                high_indices_idx[:, 4],
            ] = high_values_idx
            cnt += indices_len

        # remove padding
        high_new = high_new[
            :,
            :,
            :,
            padding_size : high_new.size(3) - padding_size,
            padding_size : high_new.size(4) - padding_size,
            padding_size : high_new.size(5) - padding_size,
        ]
        highs_recon.append(high_new)

    return highs_recon


def padding_high_indices(batch_size, highs_indices):
    batch_size_pad = torch.arange(batch_size, device=highs_indices.device)
    for _ in range(3):  # result in B * 1 * 1 * 1
        batch_size_pad = batch_size_pad.unsqueeze(1)
    batch_size_pad = batch_size_pad.repeat(
        1, highs_indices.size(0), highs_indices.size(1), 1
    )  # B * N * O * 1 (N = 46^3)
    highs_indices = highs_indices.unsqueeze(0).repeat(
        batch_size, 1, 1, 1
    )  # B * N * O * 4 (N = 46^3)
    highs_indices = torch.cat((batch_size_pad, highs_indices), dim=-1)
    return highs_indices


def extract_wavelet_coefficients(
    data_item, spatial_shapes, max_depth, keep_level, device=None
):
    # data_item : (N, 1 + 7 * D, low, low, low)

    low = data_item[:, 0:1]
    highs = []
    for j in range(max_depth):
        spatial_shape = spatial_shapes[j + 1]
        if j < keep_level:
            high = np.zeros(
                (1, 1, 7, spatial_shape[0], spatial_shape[1], spatial_shape[2])
            )
        else:
            high = data_item[
                :, (j - keep_level) * 7 + 1 : (j - keep_level + 1) * 7 + 1
            ]  # extract coefficients
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


def extract_highs_from_colors(high_indices, high_values, max_depth, shape_list):
    cnt = 0
    highs_recon = []
    for idx in range(max_depth):
        order_expand = (max_depth - 1) - idx
        indices_len = 7 * ((2**order_expand) ** 3)
        high_values_idx = high_values[:, cnt : cnt + indices_len]
        high_indices_idx = high_indices[:, cnt : cnt + indices_len]

        padding_size = (
            shape_list[-1][0] * (2**order_expand) - shape_list[idx + 1][0]
        ) // 2
        high_new = torch.zeros(
            (
                1,
                7,
                shape_list[idx + 1][0] + padding_size * 2,
                shape_list[idx + 1][1] + padding_size * 2,
                shape_list[idx + 1][2] + padding_size * 2,
                3,
            )
        ).to(high_indices_idx.device)

        ## reshape to fit the shape
        high_indices_idx = high_indices_idx.reshape((-1, 4)).long()
        high_values_idx = high_values_idx.reshape((-1, 3))

        # set the values
        high_new[
            0,
            high_indices_idx[:, 0],
            high_indices_idx[:, 1],
            high_indices_idx[:, 2],
            high_indices_idx[:, 3],
            :,
        ] = high_values_idx

        high_new = high_new[
            :,
            :,
            padding_size : high_new.size(2) - padding_size,
            padding_size : high_new.size(3) - padding_size,
            padding_size : high_new.size(4) - padding_size,
            :,
        ]  # remove padding
        high_new = torch.permute(high_new, (0, 5, 1, 2, 3, 4))
        highs_recon.append(high_new)
        cnt += indices_len
    return highs_recon


def extract_highs_from_values(high_indices, high_values, max_depth, shape_list):
    cnt = 0
    highs_recon = []
    for idx in range(max_depth):
        order_expand = (max_depth - 1) - idx
        indices_len = 7 * ((2**order_expand) ** 3)
        high_values_idx = high_values[:, cnt : cnt + indices_len]
        high_indices_idx = high_indices[:, cnt : cnt + indices_len]

        padding_size = (
            shape_list[-1][0] * (2**order_expand) - shape_list[idx + 1][0]
        ) // 2
        high_new = torch.zeros(
            (
                1,
                1,
                7,
                shape_list[idx + 1][0] + padding_size * 2,
                shape_list[idx + 1][1] + padding_size * 2,
                shape_list[idx + 1][2] + padding_size * 2,
            )
        ).to(high_indices_idx.device)

        ## reshape to fit the shape
        high_indices_idx = high_indices_idx.reshape((-1, 4)).long()
        high_values_idx = high_values_idx.reshape((-1))

        # set the values
        high_new[
            0,
            0,
            high_indices_idx[:, 0],
            high_indices_idx[:, 1],
            high_indices_idx[:, 2],
            high_indices_idx[:, 3],
        ] = high_values_idx

        high_new = high_new[
            :,
            :,
            :,
            padding_size : high_new.size(3) - padding_size,
            padding_size : high_new.size(4) - padding_size,
            padding_size : high_new.size(5) - padding_size,
        ]  # remove padding
        highs_recon.append(high_new)
        cnt += indices_len
    return highs_recon


def create_coordinates(resolution, space_range, channel_dim=7):
    channels_samples = np.linspace(0, channel_dim - 1, channel_dim)
    dimensions_samples = np.linspace(space_range[0], space_range[1], resolution)
    ch, x, y, z = np.meshgrid(
        channels_samples, dimensions_samples, dimensions_samples, dimensions_samples
    )
    ch, x, y, z = (
        ch[:, :, :, :, np.newaxis],
        x[:, :, :, :, np.newaxis],
        y[:, :, :, :, np.newaxis],
        z[:, :, :, :, np.newaxis],
    )
    coordinates = np.concatenate((ch, x, y, z), axis=4)
    coordinates = coordinates.reshape((-1, 4))
    coordinates = torch.from_numpy(coordinates).long()
    return coordinates


def extract_full_indices(device, max_depth, shape_list):
    highs_full_indices = []
    highs_full_indices_last = create_coordinates(
        shape_list[-1][0], space_range=(0, shape_list[-1][0] - 1), channel_dim=1
    ).to(device)
    for idx in range(max_depth):
        order_expand = (max_depth - 1) - idx
        highs_full_indices_idx = compute_level_indices(
            highs_full_indices_last, order_expand=order_expand
        )
        highs_full_indices_idx = highs_full_indices_idx.reshape(
            (-1, 7 * ((2**order_expand) ** 3), 4)
        )
        highs_full_indices.append(highs_full_indices_idx)
    highs_full_indices = torch.cat(highs_full_indices, dim=1)
    return highs_full_indices


def compute_level_indices(indices_keep_last, order_expand):
    indices_grid = create_coordinates(
        2**order_expand, (0, 2**order_expand - 1), channel_dim=7
    ).to(indices_keep_last.device)
    mutiplier = (
        torch.from_numpy(
            np.array([1, 2**order_expand, 2**order_expand, 2**order_expand])
        )
        .unsqueeze(0)
        .to(indices_keep_last.device)
    )
    indices_keep = (
        indices_grid.unsqueeze(0) + indices_keep_last.unsqueeze(1) * mutiplier
    )
    return indices_keep

def unique(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim,
        sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    index = index.sort().values
    return unique, inverse, counts, index

def extract_unique_indices(high_last, point_num):
    indices_keep_last = multi_dim_argsort(torch.abs(high_last)[0, 0], descending=True)
    _, _, _, idx =unique(indices_keep_last[1:], dim=0)
    indices_keep_last = indices_keep_last[idx, 1:]
    indices_keep_last = indices_keep_last[:point_num]
    indices_keep_last = torch.cat((torch.zeros((indices_keep_last.size(0), 1), device=indices_keep_last.device), indices_keep_last),
                                  dim=1).long()  ## unique last indices
    return indices_keep_last

# v_sdf: (256,256,256) ndarray
# return: low, highs
# low: (1, 1, 46, 46, 46)
# high: [(1,1,7,136,136,136), (1,1,7,76,76,76), (1,1,7,46,46,46)]
def sdf2wavelets(v_sdf, v_max_depth):
    module = DWTForward3d(J=v_max_depth, wave="bior6.8", mode="constant")
    low, highs = module(v_sdf)
    return low, highs

# v_sdf: (1,64,46,46,46) torch tensor
# return: low, highs
# low: (1, 1, 46, 46, 46)
# high: [(1,1,7,136,136,136), (1,1,7,76,76,76), (1,1,7,46,46,46)]
def wavelet_vol_to_sdf(v_vol, v_max_depth, v_keep_level, shape_list):
    data = WaveletData(shape_list,
                    max_depth=v_max_depth,
                    output_stage=v_max_depth-v_keep_level,
                    wavelet_volume=v_vol,
                    )
    low,high = data.convert_low_highs()
    return wavelets2sdf(low,high, v_max_depth)

# lows: (1, 1, 46, 46, 46)
# highs_reconstructed: [(1,1,7,136,136,136), (1,1,7,76,76,76), (1,1,7,46,46,46)]
# return: v_sdf
# v_sdf: (256,256,256)
def wavelets2sdf(lows, highs_reconstructed, max_depth):
    module = DWTInverse3d(max_depth, "bior6.8","constant")
    reconstructed_sdf = module((lows, highs_reconstructed))
    return reconstructed_sdf

# highs: [(1,1,7,136,136,136), (1,1,7,76,76,76), (1,1,7,46,46,46)]
# return: high_indices, high_values
# high_values: (16384,4)
# high_indices: (16384,511,4)
def wavelets2compactwavelets(highs, max_depth, point_num, v_is_compact, v_keep_level):
    high_last = highs[-1]
    indices_keep_last = extract_unique_indices(high_last[0:1], point_num=point_num)
    print(f"size of indices : {indices_keep_last.size(0)}")

    indices_keep_collect = compute_level_indices(indices_keep_last, order_expand=0)
    indices_keep_collect_reshaped = indices_keep_collect.reshape((-1, 4))

    last_high_value = high_last[0, 0, indices_keep_collect_reshaped[:, 0], indices_keep_collect_reshaped[:, 1],
    indices_keep_collect_reshaped[:, 2], indices_keep_collect_reshaped[:, 3]]
    last_high_value = last_high_value.reshape((-1, 7)) ## done converting the last layer

    high_values = []
    high_indices = []
    for idx, high in enumerate(highs[:-1]):
        order_expand = (max_depth - 1) - idx
        padding_size = (high_last.size(3) * (2 ** order_expand) - high.size(3)) // 2
        indices_keep = compute_level_indices(indices_keep_last, order_expand)
        indices_keep_reshape = indices_keep.reshape((-1, 4))
        print(f"padding size : {padding_size}")

        ### high padded
        high_padded = torch.nn.functional.pad(high, (padding_size, padding_size, padding_size,
                                                padding_size, padding_size, padding_size), 'constant', 0)
        high_value = high_padded[0, 0, indices_keep_reshape[:, 0], indices_keep_reshape[:, 1], indices_keep_reshape[:, 2], indices_keep_reshape[:, 3]]
        high_value = high_value.reshape((-1, 7 * ((2 ** order_expand) ** 3)))
        high_values.append(high_value)
        high_indices.append(indices_keep.reshape((-1, 7 * ((2 ** order_expand) ** 3), 4)))

    high_values.append(last_high_value)
    high_indices.append(indices_keep_collect)
    high_values = torch.cat(high_values, dim=1)
    high_indices = torch.cat(high_indices, dim=1).long()
    if v_is_compact:
        keep_level = v_keep_level
        order_expand = max_depth - keep_level
        keep_dim = (2 ** 3) ** order_expand - 1
        high_values = high_values[:, -keep_dim:]
        return indices_keep_last, high_values
    else:
        return high_indices, high_values

# lows: (1, 1, 46, 46, 46) torch  tensor
# high_values: (1,16384,4) torch  tensor
# high_indices: (1,16384,511,4) torch  tensor
# return: high_indices, high_values (torch  tensor)
# highs_recon: [(1,1,7,136,136,136), (1,1,7,76,76,76), (1,1,7,46,46,46)]
def compactwavelets2wavelets(lows, high_indices, high_values, max_depth, shape_list,keep_level=2):
    # highs_recon = extract_highs_from_values(high_indices, high_values, max_depth, shape_list)
    data = WaveletData(shape_list,
                       max_depth=max_depth,
                       output_stage=max_depth-keep_level,
                       low=lows,
                       highs_values=high_values,
                       highs_indices=high_indices,
                       )
    return data.convert_low_highs()

# v_sdf: (256,256,256)
# v_filename: output obj file
def to_mesh(v_sdf, v_filename):
    v, f = mcubes.marching_cubes(v_sdf, 0.0)
    mcubes.export_obj(v, f, v_filename)
    pass

def udf2pc(udf, threshold=0.02):
    resolution=udf.shape[0]
    coords = np.stack(np.meshgrid(np.arange(resolution),np.arange(resolution),np.arange(resolution),indexing="ij"), axis=3)

    mask = np.abs(udf) < threshold
    coords = coords[mask]

    return coords

def export_points(v_points, file):
    pc = trimesh.PointCloud(v_points)
    pc.export(str(file))