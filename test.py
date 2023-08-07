import os

def delete_files(target_folder, target_file):
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            if file == target_file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f'Removed file: {file_path}')

# Call the function with the path to your target folder and the target file name
delete_files('/home/user/xuxiao/LAM/datasets', 'seg_3.wav')
