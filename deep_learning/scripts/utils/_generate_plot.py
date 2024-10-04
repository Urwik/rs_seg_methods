import sys
import pandas as pd
import numpy as np
sys.path.append("/home/fran/workspaces/nn_ws/binary_segmentation")

from engines.utils.csv_parser import csvTestStruct
from engines.utils.graphics_plotter import GraphicsPlotter


if __name__ == '__main__':

    MODEL_NAME = ''
    X_LABEL = 'FEATURES'
    Y_LABEL = 'MIOU'
    HUE_LABEL = 'MODEL'

    plotter = GraphicsPlotter()
    plotter.set_source_file("/home/fran/workspaces/nn_ws/binary_segmentation/experiments/test.csv")
    
    plotter.filter('DATASET_DIR', 'test', negative=True)
    plotter.filter('DATASET_DIR', '00', negative=True)
    plotter.filter('DATASET_DIR', '01', negative=True)
    plotter.filter('DATASET_DIR', '02', negative=True)
    plotter.filter('DATASET_DIR', '03', negative=True)
    
    
    if MODEL_NAME != '':
        plotter.filter('MODEL', MODEL_NAME)

    # CALCULA LA MODA DE LOS F1_SCORES AGRUPADOS POR DATASET_DIR Y FEATURES
    # mode_f1_scores = plotter.df.groupby([X_LABEL, HUE_LABEL])[Y_LABEL].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).reset_index()

    #############################
    # X VALUES
    #############################
    # x = plotter.df[X_LABEL].to_numpy()
    # a, x = np.unique(x, return_inverse=True)
    # x = plotter.df['BEST_EPOCH'].to_numpy() * plotter.df['EPOCH_DURATION'].to_numpy() 
    # x = mode_f1_scores[X_LABEL].to_numpy()
    
    #############################
    # Y VALUES
    #############################
    # y = plotter.df[Y_LABEL].to_numpy()
    # y = mode_f1_scores.to_numpy()
    # y = mode_f1_scores[Y_LABEL].to_numpy()

    #############################
    # HUE VALUES
    ################
    # hue = plotter.df[HUE_LABEL].to_numpy()
    # a, hue = np.unique(hue, return_inverse=True)
    # hue = mode_f1_scores[HUE_LABEL].to_numpy()

    #############################
    # SET PLOT VALUES
    #############################
    plotter.set_x_label(X_LABEL)
    plotter.set_y_label(Y_LABEL)
    plotter.set_hue_label(HUE_LABEL)

    # plotter.set_x(x)
    # plotter.set_y(y)
    # plotter.set_hue(hue)


    #############################
    # SET PLOT TITLES
    #############################
    # plotter.set_x_title(X_LABEL)
    # plotter.set_y_title(Y_LABEL)
    # plotter.set_title(f'{MODEL_NAME}')


    #############################
    # SET PLOT STYLE
    #############################
    # plotter.set_font_size(12)
    # plotter.set_size('large')
    # plotter.y_lim(0, 1)
    # plotter.set_y_percent_formatter()
    plotter.set_color_palette('viridis_r')
    # plotter.set_size((4,7))
    # plotter.x_lim(0.5,1)
    # plotter.y_lim(-0.5,1.5)



    #############################
    # PLOT
    #############################
    # plotter.plot()
    # plotter.scatter(dinamic_size=True)
    # plotter.barplot()
    # plotter.time_to_achieve_best_label_kde('MIOU')
    # plotter.time_to_achieve_best_label_scatter('MIOU')
    # plotter.threshold_method_analysis()
    # plotter.hex_efficiency('MIOU')
    # plotter.feat_vs_metric(horizontal=True)
    # plotter.voxel_analysis()
    # plotter.inference_time()
    plotter.feat_vs_metric_roc_pr()
    # plotter.train_efficiency()
    # plotter.inference_efficiency()
    # plotter.feat_vs_metric_orto_crossed_roc_pr()

    # plotter.threshold_estimator()
    
    # plotter.barCombined()
    # plotter.plot3D()
    # plotter.set_legend()
    # plotter.image_name = 'feat_vs_metric_roc_pr_crossed'
    # plotter.save_fig_as_image(format='png')
    # plotter.save_fig_as_image(format='eps')
    
    # plotter.save_fig_as_image()
    
    # plotter.save_plt_as_image()
    # plotter.show()