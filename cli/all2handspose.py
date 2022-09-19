import click
import os
import glob
import pandas as pd


@click.command()
@click.argument("source_folder", type=click.Path(exists=True))
@click.argument("trans_folder", type=click.Path(exists=True))
def folder2folder(source_folder: str, trans_folder: str):
    """
    Transformation all csv format file to hands+pose csv format
    :param source_folder: Path
    :param trans_folder: Path
    :return:
    """
    list_path = glob.glob(source_folder + "/*")
    len_path =  len(list_path)
    for i, path in enumerate(list_path):
        trans_file = trans_folder + "/" + os.path.split(path)[-1]
        print(i+1, "of", len_path)
        print("source_file:", path)
        print("trans_file:", trans_file)
        if os.path.exists(trans_file):
            print("exist the trans_file")
        else:
            csv2csv(path, trans_file)
        print("----")
    print(len_path, "tranformated files")


def csv2csv(source_file: str, trans_file: str):
    df = pd.read_csv(source_file)
    list_col = []
    for n in df.columns:
        if "face_"==n[:5]:
            list_col.append(n)
    df_trans = df.drop(list_col, axis=1)
    df_trans.to_csv(trans_file, index=False)


if __name__ == '__main__':
    folder2folder()