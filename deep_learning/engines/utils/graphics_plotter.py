import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import PercentFormatter, MultipleLocator
import matplotlib.cm as cm


import seaborn as sns

import numpy as np
import pandas as pd
from .csv_parser import csvTestStruct


ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_END = "\033[0m"


class GraphicsPlotter():
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        # self.df = self._get_data()
        self.x = None
        self.y = None
        self.hue = None
        self.x_label = None
        self.y_label = None
        self.hue_label = None
        self.image_name = 'seaborn_plot.png'
        self.color_palette = 'viridis'

    def _get_data(self):
        pd_csv = pd.read_csv(csvTestStruct().output_file)
        return pd_csv

    # ------------------------------
    # UTILS
    # ------------------------------
    def filter(self, label, value, negative=False):
        print("Filtering by ", label)

        if label in self.df.columns:
            if type(value) == list:
                self.df = self.df[self.df[label].apply(lambda x: any(str(item) in x for item in value))]
            else:
                if negative == False:
                    self.df = self.df[self.df[label] == value]
                else:
                    self.df = self.df[self.df[label] != value]

        else: 
            print(f"{ANSI_YELLOW}ELEMENT: {label} not found in the dataframe header{ANSI_END}")

    def sort(self, label):
        if label in self.df.columns:
            self.df = self.df.sort_values(by=[label])
        else:
            print(f"{ANSI_YELLOW}ELEMENT: {label} not found in the dataframe header{ANSI_END}")
    
    def save_fig_as_image(self, path=None, format='png'):
        self.fig = self.ax.figure
        
        if path != None:
            self.fig.savefig(path, bbox_inches='tight', format=format)
        else:
            self.fig.savefig(f'/home/fran/workspaces/nn_ws/binary_segmentation/experiments/images/{self.image_name}.{format}', bbox_inches='tight' , format=format)
        
        plt.close()

    def save_plt_as_image(self, path=None):
        self.set_legend()
        if path != None:
            plt.savefig(path)
        else:
            plt.savefig(f'/home/fran/workspaces/nn_ws/binary_segmentation/experiments/{self.image_name}')
        
        plt.close()
   
    def _to_df(self):
        if isinstance(self.x, np.ndarray) and isinstance(self.y, np.ndarray) and isinstance(self.hue, np.ndarray):
            
            self.df = pd.DataFrame({
                self.x_label: self.x,
                self.y_label: self.y,
                self.hue_label: self.hue
            })
   
    # ------------------------------
    # SETTERS
    # ------------------------------
    
    def set_source_file(self, file):
        self.df = pd.read_csv(file)
        
    def set_x_label(self, label):
        if label in self.df.columns:
            self.x_label = label
        else:
            print(f"{ANSI_YELLOW}ELEMENT: {label} not found in the dataframe header{ANSI_END}")

    def set_y_label(self, label):
        if label in self.df.columns:
            self.y_label = label
        else:
            print(f"{ANSI_YELLOW}ELEMENT: {label} not found in the dataframe header{ANSI_END}")

    def set_hue_label(self, label):

        if label in self.df.columns:
            self.hue_label = label
        else:
            print(f"{ANSI_YELLOW}ELEMENT: {label} not found in the dataframe header{ANSI_END}")

    def set_x_title(self, title):
        plt.xlabel(title)

    def set_y_title(self, title):
        plt.ylabel(title)

    def set_title(self, title):
        plt.title(title)

    def set_x(self, array):
        self.x = array
    
    def set_y(self, array):
        self.y = array

    def set_hue(self, array):
        self.hue = array

    def x_lim(self, min, max):
        plt.xlim(min, max)
    
    def y_lim(self, min, max):
        plt.ylim(min, max)

    def _set_fig_title(self):
        self.ax.figure.suptitle('Test Metrics', fontsize=20)

    def set_legend(self):
        # plt.legend(loc='lower right')
        # plt.legend(loc='lower right', bbox_to_anchor=(1, 0.0))
        self.ax.legend(loc='lower center', bbox_to_anchor=(0.5,1))
    
    def set_y_percent_formatter(self):
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        
    def set_x_percent_formatter(self):
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))

    def set_color_palette(self, palette):
        self.color_palette = palette

    def enable_grid(self):
        # Set the major ticks interval for both x and y axes
        plt.gca().xaxis.set_major_locator(MultipleLocator(5))
        plt.gca().yaxis.set_major_locator(MultipleLocator(10))

        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))  # Tick labels interval
        plt.gca().yaxis.set_minor_locator(MultipleLocator(10))  # Tick labels interva
        plt.grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=0.1)

    def set_size(self, width, height):
        plt.figure(figsize=(width, height))

    def set_size(self, dim):
        
        if type(dim) == str:
            if dim == 'small':
                plt.figure(figsize=(3.5, 2))
            elif dim == 'medium':
                plt.figure(figsize=(5.5, 3))
            elif dim == 'large':
                plt.figure(figsize=(7.5, 4.2))
        elif type(dim) == tuple:
            plt.figure(figsize=dim)

    def set_font_size(self, size):
        plt.rc('font', size=size)          # controls default text sizes
        plt.rc('axes', titlesize=size)     # fontsize of the axes title
        plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
        plt.rc('legend', fontsize=size)    # legend fontsize
        plt.rc('figure', titlesize=size)
    # ------------------------------
    # PLOTTERS
    # ------------------------------
    def plot(self):
        # self.ax = sns.regplot(x=self.x_label, y=self.y_label, data=self.df, fit_reg=False)
        self.ax = sns.lmplot(x=self.x_label, y=self.y_label, data=self.df, fit_reg=False, hue=self.hue_label, legend=False)

        # if self.x is not None and self.y is not None and self.hue is not None: 
        #     self.ax = sns.scatterplot( x=self.x, y=self.y, hue=self.hue, s=150, palette='deep')

    def plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.jet(self.hue / float(max(self.hue)))

        ax.scatter(self.x, self.y, self.hue, c=colors, marker='o')
        ax.set_xlabel('DATASET')
        ax.set_ylabel('F1_SCORE')
        ax.set_zlabel('FEATURES')

    def scatter(self, dinamic_size=False):
        
        self.image_name = 'scatter_plot.png'
        
        category_order = ['XYZ', 'NXNYNZ', 'XYZNXNYNZ', 'XYZC', 'C']
        
        category_order = self.df[self.x_label].unique()
        self.df[self.x_label] = pd.Categorical(self.df[self.x_label], categories=category_order, ordered=True)

        
        if dinamic_size:
            for feature in self.df[self.x_label].unique():
                
                data = self.df[self.df[self.x_label] == feature]
                data = data.sort_values(by=self.y_label)
                data_size = data[self.y_label]
                self.ax = sns.scatterplot(x=self.x_label, y=self.y_label, hue=self.hue_label, data=data, size=data_size, sizes=(75, 200), legend=False, palette=self.color_palette)
        
        else:
            self.ax = sns.scatterplot(x=self.x_label, y=self.y_label, hue=self.hue_label, data=self.df, s=100, palette=self.color_palette)


    def feat_vs_metric(self, horizontal=False, method='max'):
        self.image_name = 'feat_vs_metric'

        # x = self.df['FEATURES'].unique()

        # if method == 'max':
        #     self.df = self.df.groupby('FEATURES').max()
        # elif method == 'mean':
        #     self.df = self.df.groupby('FEATURES').mean()
        # elif method == 'min':
        #     self.df = self.df.groupby('FEATURES').min()

        # y = self.df.groupby('FEATURES').mean()[self.y_label]


        if not horizontal:
            self.ax = sns.barplot(x=self.x_label, y=self.y_label, hue=self.hue_label, data=self.df, errorbar=None, palette='viridis')
        else:
            self.ax = sns.barplot(x=self.y_label, y=self.x_label, hue=self.hue_label, data=self.df, errorbar=None, palette='viridis', orient='h')
            self.set_x_title(self.y_label)
            self.set_y_title(self.x_label)
            self.set_x_percent_formatter()
            self.set_legend()
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.set_xlabel('')
            self.ax.set_ylabel('')
        
        for p in self.ax.patches:
            if not horizontal:
                if p.get_height() > 0:  # Only add label if the height is non-zero
                    self.ax.annotate(
                                text=format(p.get_height() * 100, '.2f'), 
                                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 8), 
                                textcoords = 'offset points',
                                fontsize=7)
            else:
                if p.get_width() > 0:  # Only add label if the width is non-zero
                    self.ax.annotate(
                        text=format(p.get_width() * 100, '.2f'), 
                        xy=(p.get_width(), p.get_y() + p.get_height() / 2.), 
                        ha='center', va='center', 
                        xytext=(12, 0), 
                        textcoords='offset points',
                        fontsize=8)
                    
            
        # plt.figure(figsize=(7, 4))


    def feat_vs_metric_orto_crossed_roc_pr(self, method='max'):
        
        DATASET_DIR = 'orto'
        TITLES_SIZE = 16
        ANNOTATION_SIZE = 12
        HUE_ORDER = ['PointTransformerV3', 'MinkUNet34C', 'PointNet++', 'PointNet']
        HUE_ORDER = HUE_ORDER[::-1]
        
        self.image_name = 'feat_vs_metric_orto_crossed_roc_pr'


        # ------------------------------
        # PREPARE DATA
        # ------------------------------
        df_pn = self.df[self.df['MODEL'] == 'PointNet']
     
        df_pnpp = self.df[self.df['MODEL'] == 'PointNet++']
     
        df_mink = self.df[self.df['MODEL'] == 'MinkUNet34C']
        df_mink = df_mink[df_mink['VOXEL_SIZE'] == 0.05]

        df_ptv3 = self.df[self.df['MODEL'] == 'PointTransformerV3']

        df_global = pd.concat([df_pn, df_pnpp, df_mink])
        
        df_roc = df_global[df_global['THRESHOLD_METHOD'] == 'roc']
        df_pr = df_global[df_global['THRESHOLD_METHOD'] == 'pr'] 
        
        df_roc = pd.concat([df_roc, df_ptv3])
        df_pr = pd.concat([df_pr, df_ptv3])
        
        df_roc_orto = df_roc[df_roc['DATASET_DIR'] == 'orto']
        df_pr_orto = df_pr[df_pr['DATASET_DIR'] == 'orto']
        
        df_roc_crossed = df_roc[df_roc['DATASET_DIR'] == 'crossed']
        df_pr_crossed = df_pr[df_pr['DATASET_DIR'] == 'crossed']

        df_roc_orto = df_roc_orto.groupby(['FEATURES', 'MODEL'], as_index=False).max()
        df_pr_orto = df_pr_orto.groupby(['FEATURES', 'MODEL'], as_index=False).max()
        
        df_roc_crossed = df_roc_crossed.groupby(['FEATURES', 'MODEL'], as_index=False).max()
        df_pr_crossed = df_pr_crossed.groupby(['FEATURES', 'MODEL'], as_index=False).max()
        

        # ------------------------------
        # PLOT FIGURE
        # ------------------------------
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,9), sharey=True)
        

        ax1 = sns.barplot(x='MIOU', y='FEATURES', hue='MODEL', data=df_pr_orto, errorbar=None, palette='viridis', hue_order=HUE_ORDER, orient='h', ax=ax1)
        ax2 = sns.barplot(x='MIOU', y='FEATURES', hue='MODEL', data=df_roc_orto, errorbar=None, palette='viridis', hue_order=HUE_ORDER, orient='h', ax=ax2)
        
        ax3 = sns.barplot(x='MIOU', y='FEATURES', hue='MODEL', data=df_pr_crossed, errorbar=None, palette='viridis', hue_order=HUE_ORDER, orient='h', ax=ax3)
        ax4 = sns.barplot(x='MIOU', y='FEATURES', hue='MODEL', data=df_roc_crossed, errorbar=None, palette='viridis', hue_order=HUE_ORDER, orient='h', ax=ax4)
        
        
        bar_values = [{},{},{},{}]
        
        
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            for patch in ax.patches:
                x_label = patch.get_y()  # Coordenada del eje Y
                hue_label = patch.get_height()  # Altura de la barra (correspondiente a hue_label)
                bar_values[i][(x_label, hue_label)] = patch.get_width()  # Ancho de la barra 
                
        
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            for (x_label, hue_label) in bar_values[i].keys():
                value = bar_values[i].get((x_label, hue_label), 0)
                
                if value != 0:
                    ax.annotate(
                        format(value * 100, '.2f'), 
                        xy=(value, x_label + hue_label / 2), 
                        xytext=(5, 0), 
                        textcoords='offset points', 
                        ha='left', va='center', 
                        fontsize=ANNOTATION_SIZE)
                
                
        # bar_values_ax1 = {}
        # bar_values_ax2 = {}
        # bar_values_ax3 = {}
        # bar_values_ax4 = {}


        # # Iterar sobre las barras del primer gráfico (ROC_AUC)
        # for p in ax1.patches:
        #     # Guardar el valor de la barra
        #     x_label = p.get_y()  # Coordenada del eje Y
        #     hue_label = p.get_height()  # Altura de la barra (correspondiente a hue_label)
        #     bar_values_ax1[(x_label, hue_label)] = p.get_width()  # Ancho de la barra

        # # Iterar sobre las barras del segundo gráfico (PR_AUC)
        # for p in ax2.patches:
        #     # Guardar el valor de la barra
        #     x_label = p.get_y()  # Coordenada del eje Y
        #     hue_label = p.get_height()  # Altura de la barra (correspondiente a hue_label)
        #     bar_values_ax2[(x_label, hue_label)] = p.get_width()  # Ancho de la barra


        # # Iterar sobre las barras del tercer gráfico (ROC_AUC)
        # for p in ax3.patches:
        #     # Guardar el valor de la barra
        #     x_label = p.get_y()
        #     hue_label = p.get_height()
        #     bar_values_ax3[(x_label, hue_label)] = p.get_width()
            
        # # Iterar sobre las barras del cuarto gráfico (PR_AUC)
        # for p in ax4.patches:
        #     # Guardar el valor de la barra
        #     x_label = p.get_y()
        #     hue_label = p.get_height()
        #     bar_values_ax4[(x_label, hue_label)] = p.get_width()
            

        # Comparar las barras de los dos ejes y resaltar el valor máximo
        # for (x_label, hue_label) in bar_values_ax1.keys():
        #     # Obtener los valores de ambas barras
        #     value_ax1 = bar_values_ax1.get((x_label, hue_label), 0)
        #     value_ax2 = bar_values_ax2.get((x_label, hue_label), 0)
            
        #     if value_ax1 != 0 and value_ax2 != 0:

        #         # Determinar cuál es el valor máximo
        #         if value_ax1 > value_ax2:
        #             # Resaltar la anotación en ax1 (negrita)
        #             ax1.annotate(
        #                         format(value_ax1 * 100, '.2f'), 
        #                         xy=(value_ax1, x_label + hue_label / 2), 
        #                         xytext=(5, 0), 
        #                         textcoords='offset points', 
        #                         ha='left', va='center', 
        #                         fontsize=ANNOTATION_SIZE, weight='bold')  # Negrita
        #             ax2.annotate(
        #                 format(value_ax2 * 100, '.2f'), 
        #                 xy=(value_ax2, x_label + hue_label / 2), 
        #                 xytext=(5, 0), 
        #                 textcoords='offset points', 
        #                 ha='left', va='center', 
        #                 fontsize=ANNOTATION_SIZE)  # Negrita
                
        #         elif value_ax1 < value_ax2:
        #             ax1.annotate(
        #                 format(value_ax1 * 100, '.2f'), 
        #                 xy=(value_ax1, x_label + hue_label / 2), 
        #                 xytext=(5, 0), 
        #                 textcoords='offset points', 
        #                 ha='left', va='center', 
        #                 fontsize=ANNOTATION_SIZE)  # Negrita
                    
        #             # Resaltar la anotación en ax2 (negrita)
        #             ax2.annotate(
        #                         format(value_ax2 * 100, '.2f'), 
        #                         xy=(value_ax2, x_label + hue_label / 2), 
        #                         xytext=(5, 0), 
        #                         textcoords='offset points', 
        #                         ha='left', va='center', 
        #                         fontsize=ANNOTATION_SIZE, weight='bold')  # Negrita
                
        #         elif value_ax1 == value_ax2:
        #             ax1.annotate(
        #                 format(value_ax1 * 100, '.2f'), 
        #                 xy=(value_ax1, x_label + hue_label / 2), 
        #                 xytext=(5, 0), 
        #                 textcoords='offset points', 
        #                 ha='left', va='center', 
        #                 fontsize=ANNOTATION_SIZE, weight='bold')  # Negrita
                    
        #             # Resaltar la anotación en ax2 (negrita)
        #             ax2.annotate(
        #                         format(value_ax2 * 100, '.2f'), 
        #                         xy=(value_ax2, x_label + hue_label / 2), 
        #                         xytext=(5, 0), 
        #                         textcoords='offset points', 
        #                         ha='left', va='center', 
        #                         fontsize=ANNOTATION_SIZE, weight='bold')  # Negrita

        
        ax1.set_title('PR', fontsize=TITLES_SIZE)
        ax2.set_title('ROC', fontsize=TITLES_SIZE)
        ax1.set_ylim(0.5, 1)
        ax2.set_ylim(0.5, 1)
        ax1.set_xlim(0.25, 1)
        ax2.set_xlim(0.25, 1)
        
        # Remapear los nombres de los ticks del eje Y
        new_labels = ['V+N', 'V+C', 'V', 'N', 'C']  # Define los nuevos nombres
        new_labels = new_labels[::-1]
        ax1.set_yticklabels(new_labels, fontsize=12)  # Cambia las etiquetas del eje Y para ax1
        ax2.set_yticklabels(new_labels, fontsize=12)  # Cambia las etiquetas del eje Y para ax2

        
        
        ax1.set_ylabel('')
        ax2.set_ylabel('')
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
            # Aumentar tamaño de los ticks del eje Y
        ax1.tick_params(axis='y', labelsize=TITLES_SIZE)
        ax2.tick_params(axis='y', labelsize=TITLES_SIZE)
        ax1.tick_params(axis='x', labelsize=TITLES_SIZE-2)
        ax2.tick_params(axis='x', labelsize=TITLES_SIZE-2)
        ax2.tick_params(labelleft=False)

        # Obtener los manejadores de la leyenda (solo de ax1)
        handles, labels = ax1.get_legend_handles_labels()
        

        # Invertir el orden de los manejadores y las etiquetas de la leyenda
        handles = handles[::-1]
        labels = labels[::-1]
        
        # Quitar leyenda individual de los subplots
        
        if ax1.get_legend():
            ax1.get_legend().remove()
        # ax1.legend_.remove()
        
        if ax2.get_legend():
            ax2.get_legend().remove()
        # ax2.legend_.remove()

        # Colocar la leyenda fuera de los gráficos
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, prop={'size': TITLES_SIZE})

        # Ajustar layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        ax1.autoscale()
        ax2.autoscale()
        
        ax1.xaxis.set_major_formatter(PercentFormatter(1))
        ax2.xaxis.set_major_formatter(PercentFormatter(1))

        plt.subplots_adjust(wspace=0.4)  # Aumenta el espacio entre los ejes


        # Guardar la figura
        fig.savefig(f'/home/fran/workspaces/nn_ws/binary_segmentation/experiments/images/{self.image_name}.png', bbox_inches='tight', format='png')
        fig.savefig(f'/home/fran/workspaces/nn_ws/binary_segmentation/experiments/images/{self.image_name}.eps', bbox_inches='tight', format='eps')

        

    def annotate_max_value_new(self, axes, data):
        assert isinstance(axes, list), "Input should be a list of axes"
        assert isinstance(data, pd.DataFrame), "Input should be a dataframe"
        
        yticklabels = [tick.get_text() for tick in axes[0].get_yticklabels()]
        yticks = axes[0].get_yticks()
        ytick_map = dict(zip(yticklabels, yticks))
        print(f'Y-Tick Labels: {yticklabels}')
        print(f'Y-Ticks: {yticks}')
        
        max_max_value = data['MIOU'].max()
        
        max = 0
        for i, ax in enumerate(axes):
            for feature in data['FEATURES'].unique():
                # Filter data for the current FEATURES group
                group_data = data[data['FEATURES'] == feature]
                # Find the maximum MIOU value within this group
                max_value = group_data['MIOU'].max()
                if max_value > max:
                    max = max_value


                # Iterate over the bars in the current axis
                for p in ax.patches:
                    # Check if the bar corresponds to the current FEATURES group
                    if feature in ytick_map:
                        # Get the y position of the bar
                        y_pos = p.get_y() + p.get_height() / 2
                        # Find the closest y-tick position
                        closest_y_tick = min(yticks, key=lambda y: abs(y - y_pos))
                        if ytick_map[feature] == closest_y_tick:
                            value = p.get_width()  # Get the width (value) of the bar

                            if value != 0:
                                # Annotate the bar and bold the highest value
                                ax.annotate(
                                    format(value * 100, '.2f'),
                                    xy=(value, y_pos),
                                    xytext=(5, 0),
                                    textcoords='offset points',
                                    weight='bold' if (value == max and i==1 )else 'normal',
                                    fontsize=12,
                                    ha='left', va='center', 
                                    color='black'
                                )
    
    
    def anotate_max_value_old(self,ax1,ax2):
        
        bar_values_ax1 = {}
        bar_values_ax2 = {}
        
        # Iterar sobre las barras del primer gráfico (PR_AUC)
        for p in ax1.patches:
            # Guardar el valor de la barra
            x_value = p.get_x()  # Coordenada del eje Y
            y_value = p.get_y()
            hue_label = p.get_height()  # Altura de la barra (correspondiente a hue_label)
            bar_values_ax1[(x_value, y_value, hue_label)] = p.get_width()  # Ancho de la barra
        
        # Iterar sobre las barras del segundo gráfico (ROC_AUC)
        for p in ax2.patches:
            # Guardar el valor de la barra
            x_value = p.get_x()  # Coordenada del eje Y
            y_value = p.get_y()
            hue_label = p.get_height()  # Altura de la barra (correspondiente a hue_label)
            bar_values_ax2[(x_value, y_value, hue_label)] = p.get_width()  # Ancho de la barra


        
        max_value = 0
        # Comparar las barras de los dos ejes y resaltar el valor máximo
        for (miou, feature, model) in bar_values_ax1.keys():
            # Obtener los valores de ambas barras
            value_ax1 = bar_values_ax1.get((miou, feature, model), 0)
            value_ax2 = bar_values_ax2.get((miou, feature, model), 0)
            
            if value_ax1 != 0 and value_ax2 != 0:

                # Determinar cuál es el valor máximo
                if value_ax1 > value_ax2:
                    # Resaltar la anotación en ax1 (negrita)
                    ax1.annotate(
                                format(value_ax1 * 100, '.2f'), 
                                xy=(value_ax1, feature + model / 2), 
                                xytext=(5, 0), 
                                textcoords='offset points', 
                                ha='left', va='center', 
                                fontsize=12, weight='bold')  # Negrita
                    ax2.annotate(
                        format(value_ax2 * 100, '.2f'), 
                        xy=(value_ax2, feature + model / 2), 
                        xytext=(5, 0), 
                        textcoords='offset points', 
                        ha='left', va='center', 
                        fontsize=12)  # Negrita
                
                elif value_ax1 < value_ax2:
                    ax1.annotate(
                        format(value_ax1 * 100, '.2f'), 
                        xy=(value_ax1, feature + model / 2), 
                        xytext=(5, 0), 
                        textcoords='offset points', 
                        ha='left', va='center', 
                        fontsize=12)  # Negrita
                    
                    # Resaltar la anotación en ax2 (negrita)
                    ax2.annotate(
                                format(value_ax2 * 100, '.2f'), 
                                xy=(value_ax2, feature + model / 2), 
                                xytext=(5, 0), 
                                textcoords='offset points', 
                                ha='left', va='center', 
                                fontsize=12, weight='bold')  # Negrita
                
                elif value_ax1 == value_ax2:
                    ax1.annotate(
                        format(value_ax1 * 100, '.2f'), 
                        xy=(value_ax1, feature + model / 2), 
                        xytext=(5, 0), 
                        textcoords='offset points', 
                        ha='left', va='center', 
                        fontsize=12, weight='bold')  # Negrita
                    
                    # Resaltar la anotación en ax2 (negrita)
                    ax2.annotate(
                                format(value_ax2 * 100, '.2f'), 
                                xy=(value_ax2, feature + model / 2), 
                                xytext=(5, 0), 
                                textcoords='offset points', 
                                ha='left', va='center', 
                                fontsize=12, weight='bold')  # Negrita

    def feat_vs_metric_roc_pr_thesis(self, method='max'):
        
        DATASET_TYPE = 'crossed'
        
        
        
        
        TITLES_SIZE = 16
        ANNOTATION_SIZE = 12
        
        # Map model names to display names
        MODEL_NAME_MAP = {
            'PointNetBinSeg': 'PointNet',
            'PointNet2BinSeg': 'PointNet++',
            'PointTrasnformerV3': 'PointTransformerV3',
            'MinkUNet34C': 'MinkUNet34C'
        }
        
        HUE_ORDER = ['PointTransformerV3', 'MinkUNet34C', 'PointNet++', 'PointNet']
        HUE_ORDER = HUE_ORDER[::-1]
        
        
        self.image_name = 'feat_vs_metric_roc_pr' +'_' + DATASET_TYPE

        # ------------------------------
        # PREPARE DATA
        # ------------------------------
        DATASET_ID = 'arvc_truss/test/' + DATASET_TYPE
        self.df = self.df[self.df['DATASET_DIR'] == DATASET_ID]
        
        # Rename models to display names
        self.df['MODEL'] = self.df['MODEL'].replace(MODEL_NAME_MAP)
        
        print(f"\nDataset filtered for: {DATASET_ID}")
        print(f"Total rows after dataset filter: {len(self.df)}")
        print(f"Unique models in data: {self.df['MODEL'].unique()}")
        print(f"Unique features in data: {self.df['FEATURES'].unique()}")
     
        df_pr = self.df[self.df['THRESHOLD_METHOD'] == 'pr'] 
        df_roc = self.df[self.df['THRESHOLD_METHOD'] == 'roc']
        
        print(f"PR rows: {len(df_pr)}, Models: {df_pr['MODEL'].unique()}")
        print(f"ROC rows: {len(df_roc)}, Models: {df_roc['MODEL'].unique()}")


        # ------------------------------
        # PLOT FIGURE
        # ------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.27, 4), sharey=True)

        ax1 = sns.barplot(x='MIOU', y='FEATURES', hue='MODEL', data=df_pr, errorbar=None, palette='viridis', hue_order=HUE_ORDER, orient='h', ax=ax1)
        
        ax2 = sns.barplot(x='MIOU', y='FEATURES', hue='MODEL', data=df_roc, errorbar=None, palette='viridis', hue_order=HUE_ORDER, orient='h', ax=ax2)

        axes = [ax1, ax2]
        data = pd.concat([df_pr, df_roc])
        
        
        self.annotate_max_value_new(axes=axes, data=data)

        ax1.set_title('PR', fontsize=TITLES_SIZE)
        ax2.set_title('ROC', fontsize=TITLES_SIZE)
        ax1.set_ylim(0.5, 1)
        ax2.set_ylim(0.5, 1)
        ax1.set_xlim(0.25, 1)
        ax2.set_xlim(0.25, 1)
        
        # Remapear los nombres de los ticks del eje Y
        new_labels = ['V+N', 'V+C', 'V', 'N', 'C']  # Define los nuevos nombres
        new_labels = new_labels[::-1]
        ax1.set_yticklabels(new_labels, fontsize=12)  # Cambia las etiquetas del eje Y para ax1
        ax2.set_yticklabels(new_labels, fontsize=12)  # Cambia las etiquetas del eje Y para ax2

        ax1.set_ylabel('')
        ax2.set_ylabel('')
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Aumentar tamaño de los ticks del eje Y    
        ax1.tick_params(axis='y', labelsize=TITLES_SIZE)
        ax2.tick_params(axis='y', labelsize=TITLES_SIZE)
        ax1.tick_params(axis='x', labelsize=TITLES_SIZE-2)
        ax2.tick_params(axis='x', labelsize=TITLES_SIZE-2)
        ax2.tick_params(labelleft=False)

        # Obtener los manejadores de la leyenda (solo de ax1)
        handles, labels = ax1.get_legend_handles_labels()
        

        # Invertir el orden de los manejadores y las etiquetas de la leyenda
        handles = handles[::-1]
        labels = labels[::-1]
        
        # Quitar leyenda individual de los subplots
        
        if ax1.get_legend():
            ax1.get_legend().remove()
        # ax1.legend_.remove()
        
        if ax2.get_legend():
            ax2.get_legend().remove()
        # ax2.legend_.remove()

        # Colocar la leyenda fuera de los gráficos
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, prop={'size': TITLES_SIZE})

        # Ajustar layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        ax1.autoscale()
        ax2.autoscale()
        
        ax1.xaxis.set_major_formatter(PercentFormatter(1))
        ax2.xaxis.set_major_formatter(PercentFormatter(1))

        plt.subplots_adjust(wspace=0.1)  # Aumenta el espacio entre los ejes

        print(f'Figure saved as {self.image_name}.png')
        print(f'Figure saved as {self.image_name}.eps')
        # Guardar la figura
        fig.savefig(f'/home/fran/workspaces/arvc_ws/src/segmentation_pkgs/rs_seg_methods_(cmes)/deep_learning/results/thesis/images/{self.image_name}.png', bbox_inches='tight', format='png')
        fig.savefig(f'/home/fran/workspaces/arvc_ws/src/segmentation_pkgs/rs_seg_methods_(cmes)/deep_learning/results/thesis/images/{self.image_name}.eps', bbox_inches='tight', format='eps')

    def feat_vs_metric_roc_pr_thesis_vertical(self, method='max'):
        
        DATASET_TYPE = 'orthogonal'
        TEXT_SCALE = 0.8
        
        TITLES_SIZE = 16 * TEXT_SCALE
        ANNOTATION_SIZE = 12 * TEXT_SCALE
        
        # Map model names to display names
        MODEL_NAME_MAP = {
            'PointNetBinSeg': 'PointNet',
            'PointNet2BinSeg': 'PointNet++',
            'PointTrasnformerV3': 'PointTransformerV3',
            'MinkUNet34C': 'MinkUNet34C'
        }
        
        HUE_ORDER = ['PointTransformerV3', 'MinkUNet34C', 'PointNet++', 'PointNet']
        HUE_ORDER = HUE_ORDER[::-1]
        
        self.image_name = 'feat_vs_metric_roc_pr_vertical_' + DATASET_TYPE

        # ------------------------------
        # PREPARE DATA
        # ------------------------------
        DATASET_ID = 'arvc_truss/test/' + DATASET_TYPE
        self.df = self.df[self.df['DATASET_DIR'] == DATASET_ID]
        
        # Rename models to display names
        self.df['MODEL'] = self.df['MODEL'].replace(MODEL_NAME_MAP)
        
        print(f"\nDataset filtered for: {DATASET_ID}")
        print(f"Total rows after dataset filter: {len(self.df)}")
        print(f"Unique models in data: {self.df['MODEL'].unique()}")
        print(f"Unique features in data: {self.df['FEATURES'].unique()}")
     
        df_pr = self.df[self.df['THRESHOLD_METHOD'] == 'pr'] 
        df_roc = self.df[self.df['THRESHOLD_METHOD'] == 'roc']
        
        print(f"PR rows: {len(df_pr)}, Models: {df_pr['MODEL'].unique()}")
        print(f"ROC rows: {len(df_roc)}, Models: {df_roc['MODEL'].unique()}")

        # ------------------------------
        # PLOT FIGURE - VERTICAL BARS
        # ------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

        # Vertical bars: x=FEATURES, y=MIOU
        ax1 = sns.barplot(x='FEATURES', y='MIOU', hue='MODEL', data=df_pr, errorbar=None, palette='viridis', hue_order=HUE_ORDER, ax=ax1)
        
        ax2 = sns.barplot(x='FEATURES', y='MIOU', hue='MODEL', data=df_roc, errorbar=None, palette='viridis', hue_order=HUE_ORDER, ax=ax2)

        ax1.set_title('PR', fontsize=TITLES_SIZE)
        ax2.set_title('ROC', fontsize=TITLES_SIZE)
        ax1.set_ylim(0.2, 1)
        ax2.set_ylim(0.2, 1)
        
        # Remapear los nombres de los ticks del eje X
        new_labels = ['C', 'N', 'V', 'V+C', 'V+N']
        ax1.set_xticklabels(new_labels, fontsize=12 * TEXT_SCALE)
        ax2.set_xticklabels(new_labels, fontsize=12 * TEXT_SCALE)

        ax1.set_ylabel('')
        ax2.set_ylabel('')
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Aumentar tamaño de los ticks
        ax1.tick_params(axis='y', labelsize=ANNOTATION_SIZE)
        ax2.tick_params(axis='y', labelsize=ANNOTATION_SIZE)
        ax1.tick_params(axis='x', labelsize=TITLES_SIZE)
        ax2.tick_params(axis='x', labelsize=TITLES_SIZE)

        # Obtener los manejadores de la leyenda (solo de ax1)
        handles, labels = ax1.get_legend_handles_labels()
        
        # Invertir el orden de los manejadores y las etiquetas de la leyenda
        handles = handles[::-1]
        labels = labels[::-1]
        
        # Quitar leyenda individual de los subplots
        if ax1.get_legend():
            ax1.get_legend().remove()
        
        if ax2.get_legend():
            ax2.get_legend().remove()

        # Colocar la leyenda fuera de los gráficos
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, prop={'size': TITLES_SIZE*TEXT_SCALE})

        # Ajustar layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        ax1.autoscale()
        ax2.autoscale()
        
        ax1.yaxis.set_major_formatter(PercentFormatter(1))
        ax2.yaxis.set_major_formatter(PercentFormatter(1))

        plt.subplots_adjust(wspace=0.2)

        print(f'Figure saved as {self.image_name}.png')
        print(f'Figure saved as {self.image_name}.eps')
        # Guardar la figura
        fig.savefig(f'/home/fran/workspaces/arvc_ws/src/segmentation_pkgs/rs_seg_methods_(cmes)/deep_learning/results/thesis/images/{self.image_name}.png', bbox_inches='tight', format='png')
        fig.savefig(f'/home/fran/workspaces/arvc_ws/src/segmentation_pkgs/rs_seg_methods_(cmes)/deep_learning/results/thesis/images/{self.image_name}.eps', bbox_inches='tight', format='eps')

    def feat_vs_metric_roc_pr_comparison_vertical(self, method='max'):
        """
        Compare ROC and PR values in overlapping bars (ROC background, PR foreground).
        Excludes PointTransformerV3 model.
        """
        
        DATASET_TYPE = 'orthogonal'
        TEXT_SCALE = 0.8
        
        TITLES_SIZE = 16 * TEXT_SCALE
        ANNOTATION_SIZE = 12 * TEXT_SCALE
        
        # Map model names to display names
        MODEL_NAME_MAP = {
            'PointNetBinSeg': 'PointNet',
            'PointNet2BinSeg': 'PointNet++',
            'PointTrasnformerV3': 'PointTransformerV3',
            'MinkUNet34C': 'MinkUNet34C'
        }
        
        # Exclude PointTransformerV3
        HUE_ORDER = ['MinkUNet34C', 'PointNet++', 'PointNet']
        HUE_ORDER = HUE_ORDER[::-1]
        
        self.image_name = 'feat_vs_metric_roc_pr_comparison_vertical_' + DATASET_TYPE

        # ------------------------------
        # PREPARE DATA
        # ------------------------------
        DATASET_ID = 'arvc_truss/test/' + DATASET_TYPE
        self.df = self.df[self.df['DATASET_DIR'] == DATASET_ID]
        
        # Rename models to display names
        self.df['MODEL'] = self.df['MODEL'].replace(MODEL_NAME_MAP)
        
        # Separate PointTransformerV3 data
        df_ptv3 = self.df[self.df['MODEL'] == 'PointTransformerV3']
        
        # Exclude PointTransformerV3 from main dataset
        self.df = self.df[self.df['MODEL'] != 'PointTransformerV3']
        
        print(f"\nDataset filtered for: {DATASET_ID}")
        print(f"Total rows after dataset filter: {len(self.df)}")
        print(f"Unique models in data: {self.df['MODEL'].unique()}")
        print(f"Unique features in data: {self.df['FEATURES'].unique()}")
     
        df_pr = self.df[self.df['THRESHOLD_METHOD'] == 'pr'] 
        df_roc = self.df[self.df['THRESHOLD_METHOD'] == 'roc']
        
        # Get PointTransformerV3 data (use PR as the single representation)
        df_ptv3_single = df_ptv3[df_ptv3['THRESHOLD_METHOD'] == 'pr']
        
        print(f"PR rows: {len(df_pr)}, Models: {df_pr['MODEL'].unique()}")
        print(f"ROC rows: {len(df_roc)}, Models: {df_roc['MODEL'].unique()}")

        # ------------------------------
        # PLOT FIGURE - OVERLAPPING BARS
        # ------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        # Get unique features for x-axis positioning
        features = df_pr['FEATURES'].unique()
        n_features = len(features)
        n_models = len(HUE_ORDER)
        
        # Calculate bar width and positions (add space for PTv3)
        bar_width = 0.2
        group_width = bar_width * (n_models + 1)  # +1 for PointTransformerV3
        
        # Get viridis colors for each model (including PTv3)
        colors = sns.color_palette('viridis', n_models + 1)
        model_colors = {model: colors[i] for i, model in enumerate(HUE_ORDER)}
        # Add PointTransformerV3 with the last color
        model_colors['PointTransformerV3'] = colors[-1]
        
        # Plot ROC bars (background - with transparency)
        for i, model in enumerate(HUE_ORDER):
            df_roc_model = df_roc[df_roc['MODEL'] == model]
            
            # Get MIOU values for each feature
            miou_values = []
            for feature in features:
                feature_data = df_roc_model[df_roc_model['FEATURES'] == feature]
                if len(feature_data) > 0:
                    miou_values.append(feature_data['MIOU'].values[0])
                else:
                    miou_values.append(0)
            
            # Calculate x positions
            x_positions = np.arange(n_features) + i * bar_width - group_width / 2 + bar_width / 2
            
            # Plot ROC bars with transparency
            ax.bar(x_positions, miou_values, bar_width, 
                   label=f'{model} (ROC)', color=model_colors[model], 
                   alpha=0.4, edgecolor='black', linewidth=0.5)
        
        # Plot PR bars (foreground - solid)
        for i, model in enumerate(HUE_ORDER):
            df_pr_model = df_pr[df_pr['MODEL'] == model]
            
            # Get MIOU values for each feature
            miou_values = []
            for feature in features:
                feature_data = df_pr_model[df_pr_model['FEATURES'] == feature]
                if len(feature_data) > 0:
                    miou_values.append(feature_data['MIOU'].values[0])
                else:
                    miou_values.append(0)
            
            # Calculate x positions (same as ROC)
            x_positions = np.arange(n_features) + i * bar_width - group_width / 2 + bar_width / 2
            
            # Plot PR bars solid
            ax.bar(x_positions, miou_values, bar_width, 
                   label=f'{model} (PR)', color=model_colors[model], 
                   alpha=1.0, edgecolor='black', linewidth=0.8)

        # Plot PointTransformerV3 as a single solid bar (using PR values)
        if len(df_ptv3_single) > 0:
            miou_values_ptv3 = []
            for feature in features:
                feature_data = df_ptv3_single[df_ptv3_single['FEATURES'] == feature]
                if len(feature_data) > 0:
                    miou_values_ptv3.append(feature_data['MIOU'].values[0])
                else:
                    miou_values_ptv3.append(0)
            
            # Position PTv3 at the end of each group
            x_positions_ptv3 = np.arange(n_features) + n_models * bar_width - group_width / 2 + bar_width / 2
            
            # Plot PTv3 bars as single solid bars
            ax.bar(x_positions_ptv3, miou_values_ptv3, bar_width, 
                   label='PointTransformerV3', color=model_colors['PointTransformerV3'], 
                   alpha=1.0, edgecolor='black', linewidth=0.8)

        # Configure plot
        ax.set_ylim(0.2, 1)
        
        # Remapear los nombres de los ticks del eje X
        new_labels = ['C', 'N', 'V', 'V+C', 'V+N']
        ax.set_xticks(np.arange(n_features))
        ax.set_xticklabels(new_labels, fontsize=12 * TEXT_SCALE)

        ax.set_ylabel('mIoU', fontsize=TITLES_SIZE)
        ax.set_xlabel('')
        ax.set_title('ROC (transparent) vs PR (solid) Comparison', fontsize=TITLES_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Aumentar tamaño de los ticks
        ax.tick_params(axis='y', labelsize=ANNOTATION_SIZE)
        ax.tick_params(axis='x', labelsize=TITLES_SIZE)

        # Create custom legend
        handles, labels = ax.get_legend_handles_labels()
        
        # Colocar la leyenda fuera de los gráficos
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, prop={'size': ANNOTATION_SIZE})

        # Ajustar layout
        plt.tight_layout()

        ax.yaxis.set_major_formatter(PercentFormatter(1))
        # ax.grid(axis='y', alpha=0.3, linestyle='--')

        print(f'Figure saved as {self.image_name}.png')
        print(f'Figure saved as {self.image_name}.eps')
        # Guardar la figura
        fig.savefig(f'/home/fran/workspaces/arvc_ws/src/segmentation_pkgs/rs_seg_methods_(cmes)/deep_learning/results/thesis/images/{self.image_name}.png', bbox_inches='tight', format='png')
        fig.savefig(f'/home/fran/workspaces/arvc_ws/src/segmentation_pkgs/rs_seg_methods_(cmes)/deep_learning/results/thesis/images/{self.image_name}.eps', bbox_inches='tight', format='eps')

    def feat_vs_metric_roc_pr(self, method='max'):
        
        # DATASET_DIR = 'crossed'
        TITLES_SIZE = 16
        ANNOTATION_SIZE = 12
        HUE_ORDER = ['PointTransformerV3', 'MinkUNet34C', 'PointNet++', 'PointNet']
        HUE_ORDER = HUE_ORDER[::-1]
        
        assert len(self.df['DATASET_DIR'].unique()) == 1, "The dataframe should contain data for only one DATASET_DIR"
        DATASET_DIR = self.df['DATASET_DIR'].unique()[0]
        
        self.image_name = 'feat_vs_metric_roc_pr' +'_' + DATASET_DIR


        # ------------------------------
        # PREPARE DATA
        # ------------------------------
        self.df = self.df[self.df['DATASET_DIR'] == DATASET_DIR]
     
        df_pn = self.df[self.df['MODEL'] == 'PointNet']
     
        df_pnpp = self.df[self.df['MODEL'] == 'PointNet++']
     
        df_mink = self.df[self.df['MODEL'] == 'MinkUNet34C']
        df_mink = df_mink[df_mink['VOXEL_SIZE'] == 0.05]

        df_ptv3 = self.df[self.df['MODEL'] == 'PointTransformerV3']

        df_global = pd.concat([df_pn, df_pnpp, df_mink])
        
        df_pr = df_global[df_global['THRESHOLD_METHOD'] == 'pr'] 
        df_roc = df_global[df_global['THRESHOLD_METHOD'] == 'roc']
        
        df_pr = pd.concat([df_pr, df_ptv3])
        df_roc = pd.concat([df_roc, df_ptv3])

        df_pr = df_pr.groupby(['FEATURES', 'MODEL'], as_index=False).max()
        df_roc = df_roc.groupby(['FEATURES', 'MODEL'], as_index=False).max()
        
        df_pr = df_pr.drop(['EXPERIMENT_ID', 'GRID_SIZE', 'VOXEL_SIZE', 'RECALL', 'PRECISION', 'F1_SCORE', 'TERMINATION_CRITERIA', 'SCHEDULER', 'OPTIMIZER', 'ACCURACY', 'TP', 'TN', 'FP', 'FN', 'NORMALIZATION', 'DEVICE', 'SEQUENCES', 'BEST_EPOCH', 'EPOCH_DURATION', 'INFERENCE_DURATION'], axis=1)
        
        df_roc = df_roc.drop(['EXPERIMENT_ID', 'GRID_SIZE', 'VOXEL_SIZE', 'RECALL', 'PRECISION', 'F1_SCORE', 'TERMINATION_CRITERIA', 'SCHEDULER', 'OPTIMIZER', 'ACCURACY', 'TP', 'TN', 'FP', 'FN', 'NORMALIZATION', 'DEVICE', 'SEQUENCES', 'BEST_EPOCH', 'EPOCH_DURATION', 'INFERENCE_DURATION'], axis=1)


        # ------------------------------
        # PLOT FIGURE
        # ------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,9), sharey=True)

        ax1 = sns.barplot(x='MIOU', y='FEATURES', hue='MODEL', data=df_pr, errorbar=None, palette='viridis', hue_order=HUE_ORDER, orient='h', ax=ax1, estimator=np.max)
        
        ax2 = sns.barplot(x='MIOU', y='FEATURES', hue='MODEL', data=df_roc, errorbar=None, palette='viridis', hue_order=HUE_ORDER, orient='h', ax=ax2, estimator=np.max)

        axes = [ax1, ax2]
        data = pd.concat([df_pr, df_roc])
        
        
        self.annotate_max_value_new(axes=axes, data=data)

        ax1.set_title('PR', fontsize=TITLES_SIZE)
        ax2.set_title('ROC', fontsize=TITLES_SIZE)
        ax1.set_ylim(0.5, 1)
        ax2.set_ylim(0.5, 1)
        ax1.set_xlim(0.25, 1)
        ax2.set_xlim(0.25, 1)
        
        # Remapear los nombres de los ticks del eje Y
        new_labels = ['V+N', 'V+C', 'V', 'N', 'C']  # Define los nuevos nombres
        new_labels = new_labels[::-1]
        ax1.set_yticklabels(new_labels, fontsize=12)  # Cambia las etiquetas del eje Y para ax1
        ax2.set_yticklabels(new_labels, fontsize=12)  # Cambia las etiquetas del eje Y para ax2

        ax1.set_ylabel('')
        ax2.set_ylabel('')
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Aumentar tamaño de los ticks del eje Y    
        ax1.tick_params(axis='y', labelsize=TITLES_SIZE)
        ax2.tick_params(axis='y', labelsize=TITLES_SIZE)
        ax1.tick_params(axis='x', labelsize=TITLES_SIZE-2)
        ax2.tick_params(axis='x', labelsize=TITLES_SIZE-2)
        ax2.tick_params(labelleft=False)

        # Obtener los manejadores de la leyenda (solo de ax1)
        handles, labels = ax1.get_legend_handles_labels()
        

        # Invertir el orden de los manejadores y las etiquetas de la leyenda
        handles = handles[::-1]
        labels = labels[::-1]
        
        # Quitar leyenda individual de los subplots
        
        if ax1.get_legend():
            ax1.get_legend().remove()
        # ax1.legend_.remove()
        
        if ax2.get_legend():
            ax2.get_legend().remove()
        # ax2.legend_.remove()

        # Colocar la leyenda fuera de los gráficos
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, prop={'size': TITLES_SIZE})

        # Ajustar layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        ax1.autoscale()
        ax2.autoscale()
        
        ax1.xaxis.set_major_formatter(PercentFormatter(1))
        ax2.xaxis.set_major_formatter(PercentFormatter(1))

        plt.subplots_adjust(wspace=0.4)  # Aumenta el espacio entre los ejes

        print(f'Figure saved as {self.image_name}.png')
        print(f'Figure saved as {self.image_name}.eps')
        # Guardar la figura
        fig.savefig(f'/home/fran/workspaces/arvc_ws/src/segmentation_pkgs/rs_seg_methods_(cmes)/deep_learning/results/thesis/images/{self.image_name}.png', bbox_inches='tight', format='png')
        fig.savefig(f'/home/fran/workspaces/arvc_ws/src/segmentation_pkgs/rs_seg_methods_(cmes)/deep_learning/results/thesis/images/{self.image_name}.eps', bbox_inches='tight', format='eps')

    def inference_efficiency(self):
        self.image_name = 'inference_efficiency'
        
        df_pn = self.df[self.df['MODEL'] == 'PointNet']
        df_pnpp = self.df[self.df['MODEL'] == 'PointNet++']

        df_mink = self.df[self.df['MODEL'] == 'MinkUNet34C']
        df_mink = df_mink[df_mink['VOXEL_SIZE'] == 0.05]

        df_ptv3 = self.df[self.df['MODEL'] == 'PointTransformerV3']

        df_global = pd.concat([df_pn, df_pnpp, df_mink, df_ptv3])
        # df_global['INFERENCE_DURATION']  

        
        self.ax = sns.barplot(x='MODEL', y='INFERENCE_DURATION', hue='FEATURES', data=df_global, errorbar=None, estimator=np.mean)

        
        # self.ax = sns.scatterplot(x='TRAIN_DURATION', y='MIOU', hue='MODEL', data=df_global, s=100, palette='viridis', alpha=0.6)
        self.set_y_title('INFERENCE_DURATION (seconds)')
        self.set_x_title('MODEL')
        self.set_legend()
        # self.set_x_percent_formatter()
        
        self.save_fig_as_image(format='png')
        self.save_fig_as_image(format='eps')
        
    def train_efficiency(self):
        self.image_name = 'train_efficiency'
        self.df['BEST_EPOCH'] += 1
        
        df_pn = self.df[self.df['MODEL'] == 'PointNet']
     
        df_pnpp = self.df[self.df['MODEL'] == 'PointNet++']
     
        df_mink = self.df[self.df['MODEL'] == 'MinkUNet34C']
        df_mink = df_mink[df_mink['VOXEL_SIZE'] == 0.05]

        df_ptv3 = self.df[self.df['MODEL'] == 'PointTransformerV3']

        df_global = pd.concat([df_pn, df_pnpp, df_mink, df_ptv3])
        df_global['BEST_EPOCH'] += 1

        df_global['TRAIN_DURATION'] = df_global['BEST_EPOCH'].to_numpy() * df_global['EPOCH_DURATION'].to_numpy() 
        
        self.ax = sns.barplot(x='MODEL', y='TRAIN_DURATION', hue='FEATURES', data=df_global, errorbar=None, estimator=np.mean)

        
        # self.ax = sns.scatterplot(x='TRAIN_DURATION', y='MIOU', hue='MODEL', data=df_global, s=100, palette='viridis', alpha=0.6)
        self.set_y_title('TRAIN_DURATION (hours)')
        self.set_x_title('MODEL')
        self.set_legend()
        # self.set_x_percent_formatter()
        
        self.save_fig_as_image(format='png')
        self.save_fig_as_image(format='eps')
        

    def time_to_achieve_best_label_scatter(self, label):
        self.image_name = 'efficiency_scatter_plot.png'
        
        self.df['TRAIN_DURATION'] = self.df['BEST_EPOCH'].to_numpy() * self.df['EPOCH_DURATION'].to_numpy() 
        
        self.ax = sns.scatterplot(x='TRAIN_DURATION', y=label, hue='MODEL', alpha=0.6, data=self.df, s=100)
        self.ax.legend().remove()


    def time_to_achieve_best_label_kde(self, label):
        self.image_name = 'efficiency_kde_plot.png'
       
        self.df['TRAIN_DURATION'] = self.df['BEST_EPOCH'].to_numpy() * self.df['EPOCH_DURATION'].to_numpy() 
       
        filtered_df = self.df
        # filtered_df = self.df.loc[self.df[label] > 0.7]
       
        color_palette = 'viridis'
        sns.set_style("white")
        # self.ax = sns.kdeplot(x='TRAIN_DURATION', y=label, hue='MODEL', data=self.df, palette=color_palette, fill=False, alpha=0.5, levels=2, zorder=0, linewidths=2)
        # self.ax = sns.kdeplot(x='TRAIN_DURATION', y=label, hue='MODEL', data=self.df, palette=color_palette, fill=True, alpha=0.8, thresh=0.65, levels=2, zorder=1)
        # self.ax = sns.kdeplot(x='TRAIN_DURATION', y=label, hue='MODEL', data=filtered_df, palette=color_palette, fill=True, alpha=0.8, thresh=0.5, levels=4, zorder=1)
        self.ax = sns.kdeplot(x='TRAIN_DURATION', y=label, hue='MODEL', data=filtered_df, palette=color_palette, fill=True, alpha=0.8, thresh=0.2, levels=3)

        
        self.ax.legend(title='', loc='upper right', bbox_to_anchor=(1, 1.2))    

        # ax = plt.gca()  # Get the current Axes instance

        # # Setting minor locators
        # ax.xaxis.set_minor_locator(MultipleLocator(0.5)) 
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))  # Tick labels interval
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # Tick labels interval
        # plt.xlim(0, 5)
        
        self.set_x_title('TRAIN_DURATION (hours)')
        self.set_y_title(label)
        self.set_title('TRAIN_DURATION vs ' + label) 


    def voxel_analysis(self):
        
        self.image_name = 'voxel_analysis.png'        
        df = self.df[self.df['MODEL'] == 'MinkUNet34C']

        # BAR PLOT  
        self.ax = sns.barplot(x='VOXEL_SIZE', y='MIOU', data=df, errorbar=None, width=0.6, palette='viridis', estimator=np.mean)
        
        # SCATTER PLOT
        # # Add horizontal noise (jitter) to the 'VOXEL_SIZE' column
        # jitter_strength = 0.005  # Adjust the strength of the jitter as needed
        # df['VOXEL_SIZE_JITTERED'] = df['VOXEL_SIZE'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
        # self.ax = sns.scatterplot(x='VOXEL_SIZE_JITTERED', y='MIOU', data=df, s=100, palette='viridis', alpha=0.6)
        
        plt.savefig(f'/home/fran/workspaces/nn_ws/binary_segmentation/experiments/images/{self.image_name}.png', bbox_inches='tight', format='png')
        plt.savefig(f'/home/fran/workspaces/nn_ws/binary_segmentation/experiments/images/{self.image_name}.eps', bbox_inches='tight', format='eps')

    def inference_time(self):
        self.image_name = 'inference_time.png'
        
        # BAR PLOT
        self.ax = sns.barplot(x='MODEL', y='INFERENCE_DURATION', ci=None, estimator=np.median, data=self.df, palette='viridis')
        # self.set_y_percent_formatter()
        
        # SCATTER PLOT
        # # Add horizontal noise (jitter) to the 'VOXEL_SIZE' column
        # jitter_strength = 0.005  # Adjust the strength of the jitter as needed
        # df['VOXEL_SIZE_JITTERED'] = df['VOXEL_SIZE'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
        # self.ax = sns.scatterplot(x='VOXEL_SIZE_JITTERED', y='MIOU', data=df, s=100, palette='viridis', alpha=0.6)


    def line_plot(self):
        from matplotlib.lines import Line2D
        cmap = plt.cm.coolwarm
                
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(.5), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        
        metrics_list = ['PRECISION', 'RECALL', 'F1_SCORE', 'MIOU', 'ACCURACY']

        for metric in metrics_list:
            self.ax = sns.lineplot(x=self.x_label, y=metric, data=self.df)

        self.ax.legend(custom_lines, metrics_list)
        self.ax.set_ylim(0.4, 1)
        # create legend


    def threshold_estimator(self):
        self.image_name = 'threshold_estimator.png'
        self.filter('MODEL', 'PointTrasnformerV3', negative=True)
        self.ax = sns.barplot(x='MODEL', y='MIOU', hue='THRESHOLD_METHOD', data=self.df, palette=self.color_palette, ci='sd')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.legend(title='', loc='upper right', bbox_to_anchor=(1, 1.2))    
        self.set_y_percent_formatter()


    def hex_efficiency(self, label='MIOU'):
        from bokeh.plotting import figure, show
        from bokeh.transform import linear_cmap
        from bokeh.util.hex import hexbin

        from bokeh.io import export_png 

        self.df['TRAIN_DURATION'] = self.df['BEST_EPOCH'].to_numpy() * self.df['EPOCH_DURATION'].to_numpy() 
        self.x = self.df['TRAIN_DURATION'].to_numpy()
        self.y = self.df[label].to_numpy()

        bins = hexbin(self.x, self.y, 0.05)

        p = figure(tools="", match_aspect=True, background_fill_color='#440154')
        p.grid.visible = False

        p.hex_tile(q="q", r="r", size=0.7, line_color=None, source=bins,
                fill_color=linear_cmap('counts', 'Viridis256', 0, max(bins.counts)))

        export_png(p, filename="/home/fran/workspaces/nn_ws/binary_segmentation/experiments/images/hexplot.png")
        # show(p)


    def threshold_method_analysis(self, label='MIOU'):
        from bokeh.layouts import column
        from bokeh.plotting import figure, show
        from bokeh.sampledata.autompg import autompg
        from bokeh.transform import jitter
        from bokeh.io import export_png 
        from bokeh.models import FixedTicker


        self.df['TRAIN_DURATION'] = self.df['BEST_EPOCH'].to_numpy() * self.df['EPOCH_DURATION'].to_numpy() 
        self.x = self.df['MODEL'].to_numpy()
        self.y = self.df['TRAIN_DURATION'].to_numpy()

        p = figure(width=600, height=300, title="Years vs mpg with jittering")
        # p.xgrid.grid_line_color = None
        # p.xaxis.ticker = FixedTicker(ticks=self.df['MODEL'].unique().tolist())

        p.scatter(x=jitter(self.x, 0.4), y=self.y, size=9, alpha=0.4)

        export_png(p, filename="/home/fran/workspaces/nn_ws/binary_segmentation/experiments/images/threshold_method_analysis.png")


    def show(self):
        plt.show()




    # ------------------------------
    # DEPRECATED
    # ------------------------------
    """ def regplot(self):
        x_numpy = self.x.to_numpy()
        y_numpy = self.y.to_numpy()
        regplot = sns.regplot(x=x_numpy, y=y_numpy, fit_reg=False)
        fig = regplot.get_figure()
        fig.savefig('/home/fran/workspaces/nn_ws/binary_segmentation/experiments/seaborn_plot.png') """

    