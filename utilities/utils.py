import os
import json

def get_save_folder(args):
    # create save and log folder
    save_path = f'{args.logging_dir}'
    save_path = os.path.join(save_path, args.dataset, args.mode, str(args.lr))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    return save_path

def load_json( json_fp ):
    assert os.path.isfile( json_fp ), "Error loading JSON. File provided does not exist, cannot read: {}".format( json_fp )
    with open( json_fp, 'r' ) as f:
        json_content = json.load(f)
    return json_content