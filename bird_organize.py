import os
import shutil
import pandas as pd

def organize_cub_for_mypair(cub_root, output_root):
    """
    Organizes the CUB-200 dataset into 'train' and 'test' folders
    with class subfolders, based on train_test_split.txt.

    Args:
        cub_root (str): Path to the root directory of the CUB-200 dataset.
        output_root (str): Path to the directory where the organized dataset will be created.
    """
    train_output_dir = os.path.join(output_root, "train")
    test_output_dir = os.path.join(output_root, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    image_folder = os.path.join(cub_root, "images")
    labels_df = pd.read_csv(os.path.join(cub_root, "image_class_labels.txt"), sep=" ", header=None, names=['idx', 'label'])
    image_paths_df = pd.read_csv(os.path.join(cub_root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
    train_test_split_df = pd.read_csv(os.path.join(cub_root, "train_test_split.txt"), sep=" ", header=None, names=['idx', 'train_flag'])
    class_names_df = pd.read_csv(os.path.join(cub_root, "classes.txt"), sep=" ", header=None, names=['label_index', 'class_name'])

    merged_df = pd.merge(labels_df, image_paths_df, on='idx')
    merged_df = pd.merge(merged_df, train_test_split_df, on='idx')
    merged_df['label'] = merged_df['label'] - 1  # Adjust labels to be 0-indexed

    for index, row in merged_df.iterrows():
        image_path = row['path']
        label_index = row['label']
        train_flag = row['train_flag']
        class_name = class_names_df.iloc[label_index]['class_name']
        source_path = os.path.join(image_folder, image_path)
        image_filename = os.path.basename(image_path)

        if train_flag == 1:
            destination_folder = os.path.join(train_output_dir, class_name)
        else:
            destination_folder = os.path.join(test_output_dir, class_name)

        os.makedirs(destination_folder, exist_ok=True)
        destination_path = os.path.join(destination_folder, image_filename)
        shutil.copy(source_path, destination_path)

    print(f"Successfully organized CUB-200 dataset in: {output_root}")

if __name__ == '__main__':
    cub_root = './bird'  # Replace with the actual path to your CUB-200 dataset
    output_root = './bird_imagefolder' # Replace with the desired output path
    organize_cub_for_mypair(cub_root, output_root)

    # After running this script, you can modify your main.py to load the bird dataset like other datasets:
    # if args.task == 'bird':
    #     traindir = os.path.join('/path/to/your/organized_bird_dataset', 'train') # Point to the train folder
    #     train_data = MyPair(img_root=traindir, transform=train_transform)
    # else:
    #     traindir = os.path.join(args.root, "train")
    #     train_data = MyPair(img_root=traindir, transform=train_transform)