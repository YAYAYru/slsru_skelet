import click
import pandas as pd
from slsru_skelet.view_opencv import show_from_csv
    

@click.command()
@click.option("--path_csv", type=click.Path(exists=True))
@click.option("--skelet_part", default="hands+pose")
@click.option("--hand_mirror", default="left")
def main(path_csv: str, skelet_part: str, hand_mirror: str): 
    df = pd.read_csv(path_csv)
    show_from_csv(df)


if __name__ == '__main__':
    main()