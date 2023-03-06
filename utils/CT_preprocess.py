import SimpleITK as sitk
import numpy as np

from utils.geometry import transpose_by_cosine_matrix

RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3


def zoom(img, new_shape, new_spacing, old_spacing, origin, direction):
    img = img.swapaxes(0, 1)  # x,y,z -> y,x,z
    sitk_img = sitk.GetImageFromArray(img)  # y,x,z
    sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(old_spacing)
    sitk_img.SetDirection(direction)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_shape)
    newimage = resample.Execute(sitk_img)
    return sitk.GetArrayFromImage(newimage).swapaxes(0, 1)  # get batch_img z,y,x -> x,y,z


def min_max_normalize(img, manual_min=None, manual_max=None, eps=1.0e-3):
    img_min = np.min(img) if manual_min is None else manual_min
    img_max = np.max(img) if manual_max is None else manual_max
    img = (img - img_min) / (img_max - img_min + eps)
    return img


def z_score_normalize(img, manual_mean=None, manual_std=None):
    img_mean = np.mean(img) if manual_mean is None else manual_mean
    img_std = np.std(img) if manual_std is None else manual_std
    img = (img - img_mean) / img_std
    return img


def window_level_normalize(img, level=None, window=None, normalize=True, normalize_func='min_max_normalize'):
    img = img.copy()
    img = img.astype('float32')
    if level is not None:
        min_HU = level - window / 2
        max_HU = level + window / 2
        img = img.clip(min=min_HU, max=max_HU)
    n_func = globals()[normalize_func]
    img = n_func(img)
    if not normalize:
        img *= 255
    return img


def getNslice(img3d, slices_num, new_key_idx, old_shape, new_shape):
    """
    new_img: (width, height, channel)
    """
    img3d = img3d.copy()
    data = np.zeros((old_shape[0], old_shape[1], slices_num))
    # data = np.zeros(np.shape(new_img)[:2] + (slices_num,))

    top_num = 0
    bottom_num = 0
    if (new_key_idx - slices_num // 2) < 0:
        top_num = np.abs(new_key_idx - slices_num // 2)

    if (new_key_idx + slices_num // 2 + 1) > new_shape[2]:
        bottom_num = (new_key_idx + slices_num // 2 + 1) - new_shape[2]

    if top_num > 0 and bottom_num > 0:
        top = [img3d[:, :, 0] for _ in range(top_num)]
        top = np.stack(top, -1)
        bottom = [img3d[:, :, -1] for _ in range(bottom_num)]
        bottom = np.stack(bottom, -1)

        data[:, :, 0:top_num] = top
        data[:, :, top_num:slices_num - bottom_num] = img3d
        data[:, :, slices_num - bottom_num:] = bottom
    elif top_num > 0:
        top = [img3d[:, :, 0] for _ in range(top_num)]
        top = np.stack(top, -1)
        data[:, :, 0:top_num] = top
        data[:, :, top_num:] = img3d[:, :, 0:new_key_idx + slices_num // 2 + 1]
    elif bottom_num > 0:
        bottom = [img3d[:, :, -1] for _ in range(bottom_num)]
        bottom = np.stack(bottom, -1)
        data[:, :, 0:slices_num - bottom_num] = img3d[:, :, new_key_idx - slices_num // 2:]
        data[:, :, slices_num - bottom_num:] = bottom
    else:
        data = img3d[:, :, new_key_idx - slices_num // 2:new_key_idx + slices_num // 2 + 1]
    return data


def adjust_spacing(ori_nii, origin, old_shape, old_spacing, direction, z_spacing):
    # DeepLesion do this
    new_spacing = [z_spacing, old_spacing[1], old_spacing[2], ]
    new_shape = [old_shape[0], old_shape[1], int(old_shape[2] * (old_spacing[0] / new_spacing[0]))]
    new_nii = zoom(ori_nii, new_shape, new_spacing, old_spacing, origin, direction) if z_spacing == 3.22 else ori_nii
    return new_nii, new_shape, new_spacing


def save_nii(image_slice, save_path, origin, new_spacing, direction):
    image_slice = image_slice.swapaxes(0, 1) * 1.0
    image_slice = sitk.GetImageFromArray(image_slice)
    image_slice.SetOrigin(origin)
    image_slice.SetSpacing(new_spacing)
    image_slice.SetDirection(direction)
    sitk.WriteImage(image_slice, save_path)


def read_dicom_by_sitk(dicom_dirs, target_direction=None):
    if target_direction is None:
        target_direction = [0, 1, 0, 1, 0, 0, 0, 0, 1]
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dirs)
    reader.SetFileNames(dicom_names)
    itk_img = reader.Execute()
    vol = sitk.GetArrayFromImage(itk_img).swapaxes(0, 2)
    ori_spacing = np.array(list(itk_img.GetSpacing()))
    ori_origin = np.array(list(itk_img.GetOrigin()))
    ori_direction = np.array(itk_img.GetDirection()).reshape(3, 3)
    vol_meta = {'ori_spacing': ori_spacing, 'ori_direction': ori_direction, 'ori_origin': ori_origin}
    target_vol, target_meta = transpose_by_cosine_matrix(vol, target_direction, vol_meta, ori_direction)
    vol_meta.update(target_meta)
    vol_meta.update(dict(zip(['height', 'width', 'slice_number'], target_vol.shape)))
    return target_vol, vol_meta


def read_nii_by_sitk(path, target_direction=None):
    """

    Args:
        path: path to CT file
        target_direction: Assuming we wanna numpy coordinates (y, x, z)

    Returns:

    """
    if target_direction is None:
        target_direction = [0, 1, 0, 1, 0, 0, 0, 0, 1]
    itk_img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(itk_img).swapaxes(0, 2)
    ori_spacing = np.array(list(itk_img.GetSpacing()))
    ori_origin = np.array(list(itk_img.GetOrigin()))
    ori_direction = np.array(itk_img.GetDirection()).reshape(3, 3)
    vol_meta = {'ori_spacing': ori_spacing, 'ori_direction': ori_direction, 'ori_origin': ori_origin}
    target_vol, target_meta = transpose_by_cosine_matrix(vol, target_direction, vol_meta, ori_direction)
    vol_meta.update(target_meta)
    vol_meta.update(dict(zip(['height', 'width', 'slice_number'], target_vol.shape)))
    return target_vol, vol_meta
