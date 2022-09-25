import click
import pandas as pd
import matplotlib.pyplot as plt


@click.command()
# @click.argument("filepath", type=click.Path(exists=True))
@click.option("--path_from_csv", default=None)
@click.option("--path_to_csv", default=None)
def main(path_from_csv: str, path_to_csv: str):
    feature_names = []
    SELECT_FEATURES = "21hands"
    if SELECT_FEATURES=="21hands_25pose_xyz":
        for i in range(25):
            feature_names.append("pose_x" + str(i))
            feature_names.append("pose_y" + str(i))
            feature_names.append("pose_z" + str(i))
        for i in range(21):
            feature_names.append("lhand_x" + str(i))
            feature_names.append("lhand_y" + str(i))
            feature_names.append("lhand_z" + str(i))
        for i in range(21):
            feature_names.append("rhand_x" + str(i))
            feature_names.append("rhand_y" + str(i))
            feature_names.append("rhand_z" + str(i))    
    if SELECT_FEATURES=="21hands":
        for i in range(21):
            feature_names.append("lhand_x" + str(i))
            feature_names.append("lhand_y" + str(i))
            feature_names.append("lhand_z" + str(i))
        for i in range(21):
            feature_names.append("rhand_x" + str(i))
            feature_names.append("rhand_y" + str(i))
            feature_names.append("rhand_z" + str(i))    

    print(path_from_csv, path_to_csv)
    df_from = pd.read_csv(path_from_csv)[feature_names]
    df_to = pd.read_csv(path_to_csv)[feature_names]
    print("df_from",df_from)
    df_from.plot()
    print("df_to",df_to)
    df_to.plot()
    plt.show()
    


if __name__ == '__main__':
    main()
