def predict(input):
    # Auto-detect input type
    if input.ndim == 4:  # Image
        return model(image_batch)
    elif input.ndim == 5:  # Video
        return model(video_frames).mean(dim=1)  # Temporal average