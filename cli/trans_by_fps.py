import click

from slsru_skelet import trans_by_fps


@click.command()
# @click.argument("filepath", type=click.Path(exists=True))
@click.option("--paths_from_folder_csv", default=None)
@click.option("--path_to_folder_csv", default=None)
@click.option("--paths_from_folder_json", default=None)
@click.option("--path_to_folder_json", default=None)
@click.option("--to_fps", type=click.INT)
@click.option("--feature_names", default="hands21_pose25_xyz")
def main(paths_from_folder_csv: str, 
         path_to_folder_csv: str, 
         paths_from_folder_json: str,
         path_to_folder_json: str,
         to_fps: int,
         feature_names: str):
    if paths_from_folder_csv and path_to_folder_csv:
        trans_by_fps.folder_skelet2skelet_folder(paths_from_folder_csv, path_to_folder_csv, to_fps, feature_names)


if __name__ == '__main__':
    main()
