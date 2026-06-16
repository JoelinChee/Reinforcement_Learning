import kagglehub

# Download latest version
path = kagglehub.dataset_download("danielwe14/stereocamera-chessboard-pictures")

print("Path to dataset files:", path)