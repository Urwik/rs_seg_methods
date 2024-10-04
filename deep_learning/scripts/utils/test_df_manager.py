import sys
import pandas as pd
import numpy as np
sys.path.append("/home/arvc/Fran/workSpaces/nn_ws/binary_segmentation")

from engines.utils.csv_parser import csvTestStruct
from engines.utils.graphics_plotter import GraphicsPlotter


if __name__ == '__main__':
    df = pd.read_csv(csvTestStruct().output_file)

    df_grouped = df.groupby(['DATASET_DIR', 'FEATURES'])['MIOU'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).reset_index()
