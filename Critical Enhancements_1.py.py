def preprocess(input):
    if isinstance(input, list):  # Video frames
        return temporal_stack(frames)
    else:  # Single image
        return spatial_augment(image)