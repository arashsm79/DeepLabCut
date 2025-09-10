import deeplabcut

config = "/home/max/tmp/pr_3079_error/trimice-dlc-2021-06-22/config.yaml"
BU_SHUFFLE = 1

video = "/home/max/tmp/pr_3079_error/trimice-dlc-2021-06-22/videos/videocompressed1.mp4"

deeplabcut.analyze_videos(config=config, videos=[video], shuffle=BU_SHUFFLE, snapshot_index=1,)
