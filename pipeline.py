import pandas as pd
import numpy as np
import rpy2
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion, default_converter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold

from statsmodels.stats.multitest import multipletests

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm 
import random
r = ro.r

# 加载R的WaveICA库
ro.r('library(WaveICA)')

# 定义一个Python函数来包装R中的WaveICA函数
def waveica(data, wf="haar", batch=None, group=None, K=20, t=0.05, t2=0.05, alpha=0):
    # 将Pandas DataFrame转换为R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        print(type(data))
        data_r = ro.conversion.py2rpy(data)
        batch_r = ro.conversion.py2rpy(batch)
        group_r = ro.conversion.py2rpy(group)
    # 调用R中的WaveICA函数
    with conversion.localconverter(default_converter):
        result_r = ro.r['WaveICA'](data_r, wf=wf, batch=batch_r, group=group_r, K=K, t=t, t2=t2, alpha=alpha)
    
    # 将结果转换回Pandas DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        result = ro.conversion.rpy2py(result_r)
    return result

class BatchEffectsPipeline:
    def __init__(self) -> None:
        self.__valid_methods = ['waveica']
        

    def preprocess(self, file_name):
        self.file_name = file_name.split('/')[-1].split('.')[0]
        self.input_data = pd.read_csv(file_name)
        self.input_data['group'] = self.input_data['group'].astype(str)

        input_data_merge_order = self.input_data.sort_values('injection.order')
        input_data_merge_order.loc[input_data_merge_order['group']=='QC', 'group'] = '2'
        self.stat_order = input_data_merge_order.iloc[:,6:].astype(float)
        self.group_zong = input_data_merge_order['group'].astype(int)
        self.batch_zong = input_data_merge_order['batch']
        self.group_sample = self.group_zong[self.group_zong!=2]
        self.batch_sample = self.batch_zong[self.group_zong!=2]

        self.batch_qc = self.batch_zong[self.group_zong==2]

        self.input_data_stat_order_sample = self.stat_order[self.group_zong!=2]
        self.input_data_stat_order_QC = self.stat_order[self.group_zong==2]


    def calc_pc_dist(self, metric='euclidean', original=True, QC=False):
        if original:
            data = self.stat_order.copy()
        else:
            data = self.data_zong_processed.copy()
        group = np.array(self.group_zong)
        # PCA
        pca = PCA()
        pc_original = pca.fit_transform(data)

        # 选择特定组的样本的前三个主成分
        if not QC:
            pc_original_select = pc_original[group == 2, :3]
        else:
            pc_original_select = pc_original[group != 2, :3]

        # 计算欧几里得距离
        dist_original = pdist(pc_original_select, metric)

        # 将距离向量转换为矩阵
        dist_original_matrix = squareform(dist_original)

        # 计算距离矩阵中所有元素的平均值（排除对角线）
        n = dist_original_matrix.shape[0]
        avg_dist = np.sum(dist_original_matrix) / (n**2 - n)
        
        return avg_dist
    
    def calc_univariate(self, data, label, t_test=True, Wilcox=True, AUC=True, FDR=True, VIP=True, FC=True, comps=3):
        data = pd.DataFrame(data).reset_index(drop=True)
        label = pd.Series(label).reset_index(drop=True)
        if len(label.unique()) > 2:
            raise ValueError("label must have two levels")
        # print(label.unique())
        results = pd.DataFrame(index=data.columns, columns=['P_t.test', 'P_wilcox', 'AUC', 'VIP', 'P.FDR', 'FC'])
        
        if t_test:
            print("########### ttest ###########")
            results['P_t.test'] = [ttest_ind(data[col][label == label.unique()[0]], data[col][label == label.unique()[1]])[1] for col in data.columns]
        
        if Wilcox:
            print("########### wilcox ###########")
            results['P_wilcox'] = [mannwhitneyu(data[col][label == label.unique()[0]], data[col][label == label.unique()[1]], alternative='two-sided')[1] for col in data.columns]
        
        if AUC:
            print("########### AUC ###########")
            results['AUC'] = [roc_auc_score(label, data[col]) for col in data.columns]
        
        if VIP:
            print("########### VIP ###########")
            unique_labels = np.unique(label)
            label1 = np.where(label == unique_labels[0], unique_labels[1], unique_labels[0])

            # 将label和label1合并为一个二维数组
            label12 = np.column_stack((label, label1))

            # 创建PLS回归模型并进行交叉验证
            pls = PLSRegression(n_components=comps)
            pls.fit(data, label12)

            # 计算VIP分数
            T = pls.x_scores_
            W = pls.x_weights_
            Q = pls.y_loadings_
            p, h = W.shape
            vips = np.zeros((p,))

            s = np.diag(np.dot(np.dot(np.dot(T.T, T), Q.T), Q)).reshape(h, -1)
            total_s = np.sum(s)

            for i in range(p):
                weight = np.array([(W[i, j] / np.linalg.norm(W[:, j]))**2 for j in range(h)])
                vips[i] = np.sqrt(p * (np.dot(s.T, weight)) / total_s)
            results['VIP'] = vips
            
        if FDR:
            print("########### FDR ###########")
            with conversion.localconverter(default_converter):
                fdrtool = importr('fdrtool')

                # 将Python数组转换为R向量
                p_value_r = ro.FloatVector(results['P_t.test'])

                # 在R中调用fdrtool函数
                qval = fdrtool.fdrtool(p_value_r, statistic="pvalue", plot=False)

                results['P.FDR'] = np.array(qval.rx2('qval'))
        
        if FC:
            print("########### Fold Change ###########")
            results['FC'] = np.log(data[label == label.unique()[1]].mean() / data[label == label.unique()[0]].mean())
        
        return results
    
    def cal_QC_correlation(self, method="pearson"):
        data_before = self.input_data_stat_order_QC.copy()
        data_after = self.data_qc_processed.copy()
        # 计算相关性矩阵
        data_before_cor = data_before.T.corr(method=method)
        data_after_cor = data_after.T.corr(method=method)
        
        # 初始化结果DataFrame
        C = pd.DataFrame(columns=["var_before", "var_after", "cor_before", "cor_after"])
        
        # 填充结果DataFrame
        count = 0
        for i in range(data_before_cor.shape[0] - 1):
            for j in range(i + 1, data_before_cor.shape[1]):
                C.loc[count] = [i+1, j+1, data_before_cor.iat[i, j], data_after_cor.iat[i, j]]
                count += 1
        return C['cor_before'].mean(), C['cor_after'].mean()
        print(C['cor_before'].mean())
        print(C['cor_after'].mean())

    def cross_validation(self, original=True):
        if original:
            data = self.input_data_stat_order_sample.copy()
        else:
            data = self.data_sample_processed.copy()
        label = self.group_sample.copy()

        univariate_data = self.calc_univariate(data=data, label=label, t_test=True, Wilcox=False, AUC=False, FDR=True, VIP=True, FC=False, comps=3)
        var_select_data = univariate_data.index[(univariate_data['VIP'] > 1) & (univariate_data['P.FDR'] < 0.05)]

        selected_features = data[var_select_data]

        # 创建SVM模型，使用RBF核
        svm_model = SVC(kernel='rbf',C=1, probability=True)
        cv = KFold(n_splits=5, shuffle=True, random_state=9999)

        # 进行5折交叉验证
        scores = cross_val_score(svm_model, selected_features, label, cv=cv, scoring='roc_auc')

        # 打印交叉验证的结果
        print("Cross-validation scores:", scores)
        print("Mean score:", scores.mean())

        univariate_data_order = univariate_data.sort_values(by='VIP', ascending=False)
        auc_list = []
        print('Predictive accuracy with the same number of variables')
        for i in tqdm(range(50, 1001, 50), ncols=50):
            
            # 选择前i个变量
            var_select = univariate_data_order.index[:i]
            selected_features = data[var_select]
            
            # 创建SVM模型，使用RBF核
            svm_model = SVC(kernel='rbf', probability=True)
            
            # 进行5折交叉验证，使用AUC作为评分指标
            auc_scores = cross_val_score(svm_model, selected_features, label, cv=5, scoring='roc_auc')
            
            # 计算平均AUC并添加到列表中
            mean_auc = auc_scores.mean()
            auc_list.append(mean_auc)

        # 打印所有计算的AUC值
        if original:
            self.original_auc_list = auc_list
        else:
            self.processed_auc_list = auc_list
        return scores.mean()
        
    def plot_pca(self, original=True, fig_type='group', elev=20., azim=-160):
        if original:
            data = self.stat_order.copy()
        else:
            data = self.data_zong_processed.copy()
            
        group = self.group_zong
        batch = self.batch_zong
        # PCA
        scaler = StandardScaler()
        pca = PCA(n_components=3)
        data_scaled = scaler.fit_transform(data)
        data_pca = pca.fit_transform(data_scaled)
        proportion = pca.explained_variance_ratio_ * 100

        markers = ['o', '^', 's', 'D']
        colors = ['r', 'blue', 'black', 'green']
        if fig_type == 'group':
            # 绘制group的PCA图
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            for i, label in enumerate(["group1", "group2", "QC"]):
                ax.scatter(data_pca[group == i, 0], data_pca[group == i, 1], data_pca[group == i, 2], label=label, marker=markers[i], alpha=0.9, s=8, c=colors[i])
            ax.set_xlabel('PC1 ({:.2f}%)'.format(proportion[0]))
            ax.set_ylabel('PC2 ({:.2f}%)'.format(proportion[1]))
            ax.set_zlabel('PC3 ({:.2f}%)'.format(proportion[2]))
            ax.view_init(elev=elev, azim=azim)
            ax.legend()

            if original:
                plt.savefig("PCA_Original(group)_3D.jpeg", dpi=500)
            else:
                plt.savefig("PCA_Processed(group)_3D.jpeg", dpi=500)
            return fig
        elif fig_type == 'batch':
        # 绘制batch的PCA图
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, label in enumerate([f"batch{i+1}" for i in range(len(batch.unique()))]):
                ax.scatter(data_pca[batch == i, 0], data_pca[batch == i, 1], data_pca[batch == i, 2], label=label, marker=markers[i], alpha=0.9, s=8, c=colors[i])
            ax.set_xlabel('PC1 ({:.2f}%)'.format(proportion[0]))
            ax.set_ylabel('PC2 ({:.2f}%)'.format(proportion[1]))
            ax.set_zlabel('PC3 ({:.2f}%)'.format(proportion[2]))
            ax.view_init(elev=elev, azim=azim)
            ax.legend()
            if original:
                plt.savefig("PCA_Original(batch)_3D.jpeg", dpi=500)
            else:
                plt.savefig("PCA_Processed(batch)_3D.jpeg", dpi=500)
            # plt.close()
            return fig


    def plot_heat_map(self, original=True):
        if original:
            data = self.input_data_stat_order_QC.copy()
        else:
            data = self.data_qc_processed.copy()
        data.index = ["QC" + str(i) for i in range(1, self.input_data_stat_order_QC.shape[0]+1)]

        # 计算相关性矩阵
        cor_original = data.T.corr()

        # 创建调色板
        # my_palette = sns.color_palette("Spectral_r", 4)

        # 创建热图，并设置cbar=False来移除颜色条
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cor_original, cmap='RdYlBu_r', vmin=0.7, vmax=1, cbar=True)

        # 设置坐标轴标签
        plt.xticks(np.arange(1, len(cor_original.index), 5), cor_original.index[::5], rotation=90, fontweight='bold')
        plt.yticks(np.arange(1, len(cor_original.columns), 5), cor_original.columns[::5], fontweight='bold')

        # 保存图像
        # if original:
        #     plt.savefig("heatmap_Original.tiff", dpi=500, format='tiff')
        # else:
        #     plt.savefig("heatmap_Processed.tiff", dpi=500, format='tiff')
        # plt.close()

        return fig
    
    def plot_pair_scatter(self, selected_columns=None, original=True):
        if original:
            data = self.input_data_stat_order_QC.copy()
        else:
            data = self.data_qc_processed.copy()

        data.index = ["QC" + str(i) for i in range(1, self.input_data_stat_order_QC.shape[0]+1)]
        if selected_columns is None:
            if original:
                self.selected_columns = random.sample(list(data.index), k=3)
            selected_columns = self.selected_columns
        log_data = np.log(data.T)
        selected_data = log_data[selected_columns]
        sns.set_theme(style="ticks")
        g = sns.pairplot(selected_data, plot_kws={'s': 8})  # s控制点的大小

        # 设置字体大小
        plt.rcParams.update({'font.size': 8})

        # 保存图像
        # if original:
        #     pairplot_fig.savefig("QC_scatterplotmatrix_Original.tiff", dpi=500, format='tiff')
        # else:
        #     pairplot_fig.savefig("QC_scatterplotmatrix_Processed.tiff", dpi=500, format='tiff')

        # plt.close()
        return g.figure

    def plot_compared_auc(self):
        data_auc = pd.DataFrame({
            'auc': self.original_auc_list + self.processed_auc_list,
            'var': list(range(50, 1001, 50)) * 2,
            'group': [1] * len(self.original_auc_list) + [2] * len(self.processed_auc_list)
        })

        # 设置绘图风格
        sns.set_theme(style="whitegrid")

        # 创建绘图
        fig = plt.figure(figsize=(10, 6))
        sns.lineplot(data=data_auc, x='var', y='auc', hue='group', style='group', 
                    palette='Set1', size=1, markers=True, dashes=[(2, 2), (1, 0)])

        # 设置图例标签
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles[0:], labels=['Original', 'WaveICA'])

        # 设置坐标轴标签和标题
        plt.xlabel('# features selected', fontsize=14, fontweight='bold')
        plt.ylabel('AUC', fontsize=14, fontweight='bold')

        # 保存图像
        # plt.savefig("Comparing AUC.tiff", dpi=500, format='tiff')
        return fig
    def plot_rsd(self):
        df = self.input_data_stat_order_QC
        mean = np.mean(df)
        std_dev = np.std(df)
        rsd_origin = (std_dev / mean)

        df = self.data_qc_processed
        mean = np.mean(df)
        std_dev = np.std(df)
        rsd_processed = (std_dev / mean)

        rsd_processed = pd.DataFrame(rsd_processed).assign(source='processed')
        rsd_processed.columns = ['rsd', 'data']
        rsd_origin = pd.DataFrame(rsd_origin).assign(source='origin')
        rsd_origin.columns = ['rsd', 'data']
        rsd_df = pd.concat([rsd_origin, rsd_processed])
        fig1 = plt.figure()
        sns.boxplot(data=rsd_df, x='data', y='rsd')
        plt.title('Box plot for feature RSDs')

        # 定义划分的数值
        bins = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 1, 99999]

        # 初始化一个空的 DataFrame 来存储结果
        result = pd.DataFrame()

        grouped = rsd_df.groupby('data')

        for name, group in grouped:
            # 对每组的 total_bill 列进行划分
            group['binned'] = pd.cut(group['rsd'], bins).apply(lambda x: x.right)
            # 对划分结果进行计数统计
            stats = group['binned'].value_counts().sort_index().reset_index()
            stats.columns = ['RSD', 'Count']
            stats['RSD'] = stats['RSD'].apply(lambda x: str(100*x)+'%' if x!=99999 else 'total')
            stats['data'] = name

            stats2 = group['binned'].value_counts().sort_index().cumsum().reset_index()
            stats2.columns = ['RSD', 'CumCount']
            stats2['RSD'] = stats2['RSD']
            stats['CumCount'] = stats2['CumCount']
            result = pd.concat([result, stats])

        # 绘制条形图
        

        fig2, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x='RSD', y='Count', hue='data', data=result, ax=ax)
        lineplot = sns.lineplot(x='RSD', y='CumCount', hue='data', data=result, ax=ax, markers=True, dashes=True, style='data')
        total_count = max(result['CumCount'])
        plt.title(f'Distribution of feature RSDs (total = {total_count})')

        return fig1, fig2

    def remove_batch_effects(self, method='waveica'):
        assert method in self.__valid_methods, f'{method} is not a valid Method!' 
        data = self.stat_order
        if method == 'waveica':
            result = waveica(data, wf="haar", batch=self.batch_zong, group=self.group_zong, K=20, t=0.05, t2=0.05, alpha=0)
            tmp_df = pd.DataFrame(result['data_wave'], columns=data.columns, index=self.stat_order.index)
            # tmp_df[tmp_df<0] = 0
            self.data_zong_processed = tmp_df
            # 小于0的要设置为0

            self.data_sample_processed = self.data_zong_processed.loc[self.group_zong!=2]
            self.data_qc_processed = self.data_zong_processed.loc[self.group_zong==2]


if __name__ == '__main__':
    pipeline = BatchEffectsPipeline()
    path = './peaktable_before.csv'
    
    pipeline.preprocess(path)
    pipeline.file_name = path.split('/')[-1].split('.')[0]
    print(pipeline.calc_pc_dist())
    print(pipeline.calc_pc_dist(QC=True))
    pipeline.plot_pca()
    pipeline.cross_validation()
    pipeline.plot_heat_map()
    pipeline.plot_pair_scatter(['QC21', 'QC25', 'QC34', 'QC46', 'QC62', 'QC68'])

    pipeline.remove_batch_effects(method='waveica')

    print(pipeline.calc_pc_dist(original=False))
    print(pipeline.calc_pc_dist(original=False, QC=True))
    pipeline.plot_pca(original=False)
    pipeline.cross_validation(original=False)
    pipeline.plot_heat_map(original=False)
    pipeline.plot_pair_scatter(selected_columns=['QC21', 'QC25', 'QC34', 'QC46', 'QC62', 'QC68'], original=False)
    pipeline.plot_compared_auc()
    pipeline.cal_QC_correlation()
    
    pipeline.data_zong_processed.to_csv(f'{pipeline.file_name}_processed.csv')
# 问题：
# 1. group一般只有2个吗？
    

## 
# 上传文件
# 计算距离和QC距离，交叉验证
# 绘制PCA图像、热力图以及
