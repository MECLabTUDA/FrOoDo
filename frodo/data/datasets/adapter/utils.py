def create_mapping_dict(
    image, ood_mask=None, segmentation_mask=None, class_vector=None, **kwargs
):
    d = {"image": image}
    if ood_mask != None:
        d["ood_mask"] = ood_mask
    if segmentation_mask != None:
        d["segmentation_mask"] = segmentation_mask
    if class_vector != None:
        d["class"] = class_vector
    for k, v in kwargs.items():
        d[k] = v
    return d
