import gradio as gr
from pipeline import BatchEffectsPipeline
from datetime import datetime

pipeline_instance = BatchEffectsPipeline()

def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths

def preprocess(file_path):
    pipeline_instance.preprocess(file_path)

    pca_fig_origin_group = pipeline_instance.plot_pca(fig_type='group')
    pca_fig_origin_batch = pipeline_instance.plot_pca(fig_type='batch')
    mean_score = pipeline_instance.cross_validation()
    mean_score = round(mean_score, 4)
    heat_map_fig = pipeline_instance.plot_heat_map()
    scatter_fig = pipeline_instance.plot_pair_scatter()
    return round(pipeline_instance.calc_pc_dist(), 2), round(pipeline_instance.calc_pc_dist(QC=True), 2), pca_fig_origin_group, pca_fig_origin_batch, mean_score, heat_map_fig

def process(method):
    pipeline_instance.remove_batch_effects(method=method)

    pca_fig_after_group = pipeline_instance.plot_pca(original=False, fig_type='group')
    pca_fig_after_batch = pipeline_instance.plot_pca(original=False, fig_type='batch')
    mean_score = pipeline_instance.cross_validation(original=False)
    mean_score = round(mean_score, 4)
    heat_map_fig = pipeline_instance.plot_heat_map(original=False)
    scatter_fig = pipeline_instance.plot_pair_scatter(original=False)
    auc_compare_fig = pipeline_instance.plot_compared_auc()
    rsd_box_fig, rsd_fig = pipeline_instance.plot_rsd()
    corr_origin, corr_after = pipeline_instance.cal_QC_correlation()
    corr_origin, corr_after = round(corr_origin, 2), round(corr_after, 2)
    timestamp_for_filename = datetime.now().strftime("%Y%m%d%H%M")
    processed_path = f'.tmp/{pipeline_instance.file_name}_{method}_{timestamp_for_filename}.csv'
    pipeline_instance.data_zong_processed.to_csv(processed_path)
    return round(pipeline_instance.calc_pc_dist(original=False),2), round(pipeline_instance.calc_pc_dist(original=False, QC=True),2), pca_fig_after_group, pca_fig_after_batch, mean_score, heat_map_fig, auc_compare_fig, rsd_box_fig, rsd_fig, corr_origin, corr_after, processed_path
    

def update_pca_origin_batch_plot(elev, azim):
    return pipeline_instance.plot_pca(fig_type='batch', elev=elev, azim=azim)
def update_pca_origin_group_plot(elev, azim):
    return pipeline_instance.plot_pca(fig_type='group', elev=elev, azim=azim)

def update_pca_after_batch_plot(elev, azim):
    return pipeline_instance.plot_pca(original=False, fig_type='batch', elev=elev, azim=azim)
def update_pca_after_group_plot(elev, azim):
    return pipeline_instance.plot_pca(original=False, fig_type='group', elev=elev, azim=azim)

with gr.Blocks(title='去批次效应算法', theme=gr.themes.Soft()) as demo:
    gr.Markdown('''# 去批次效应算法网页版 
                   ### Powered by <img src="file/logo5_20221122.png" alt="替代文本" title="可选标题" width="200" height="100">
                ''')
    with gr.Row():
        process_button = gr.Button("运行去除批次效应算法", scale=1)
        method_radio = gr.Radio(["waveica"], label="算法选择", value="waveica", info="选择需要运行的批次效应算法", scale=4)
        QC_correlation_origin = gr.Textbox(label='处理前的QC相关性')
        QC_correlation_after = gr.Textbox(label='处理后的QC相关性')
    with gr.Row(equal_height=True):
        file_input = gr.File(label="待处理的csv文件", scale=1)
        # pre_process_button = gr.Button("预处理", scale=1, )
        with gr.Column(scale=1):
            pc_dist_sample_origin = gr.Textbox(label="处理前的样本主成分距离", info="计算所有样本的主成分的欧几里得距离之和", interactive=False)
            pc_dist_QC_origin = gr.Textbox(label="处理前的QC主成分距离", info="计算所有QC的主成分的欧几里得距离之和", interactive=False)
            cv_mean_score_origin = gr.Textbox(label='处理前的交叉验证得分', info="此处的得分为使用rbf核的SVM进行5折交叉验证的分类准确率", interactive=False)
        
        output_file = gr.File(label='处理后的数据', scale=1)
        with gr.Column(scale=1):
            pc_dist_sample_after = gr.Textbox(label="处理后的样本主成分距离", info="计算所有样本的主成分的欧几里得距离之和")
            pc_dist_QC_after = gr.Textbox(label="处理后的QC主成分距离", info="计算所有QC的主成分的欧几里得距离之和")
            cv_mean_score_after = gr.Textbox(label='处理前的交叉验证得分', info="此处的得分为使用rbf核的SVM进行5折交叉验证的分类准确率")
    
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=1):
                pca_batch_origin = gr.Plot(label="处理前的Batch PCA图像")
                with gr.Row():
                    pca_batch_origin_ele = gr.Slider(minimum=-90, maximum=90, value=20, step=1, label="Elevation")
                    pca_batch_origin_azi = gr.Slider(minimum=-180, maximum=180, value=-160, step=1, label="Azimuth")
            with gr.Column():
                pca_batch_after = gr.Plot(label="处理后的Batch PCA图像")
                with gr.Row():
                    pca_batch_after_ele = gr.Slider(minimum=-90, maximum=90, step=1, value=20, label="Elevation")
                    pca_batch_after_azi = gr.Slider(minimum=-180, maximum=180, step=1, value=-160, label="Azimuth")
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=1):
                pca_group_origin = gr.Plot(label="处理前的Group PCA图像")
                with gr.Row():
                    pca_group_origin_ele = gr.Slider(minimum=-90, maximum=90, value=20, step=1, label="Elevation")
                    pca_group_origin_azi = gr.Slider(minimum=-180, maximum=180, value=-160, step=1, label="Azimuth")
            with gr.Column():
                pca_group_after = gr.Plot(label="处理后的Group PCA图像")
                with gr.Row():
                    pca_group_after_ele = gr.Slider(minimum=-90, maximum=90, step=1, value=20, label="Elevation")
                    pca_group_after_azi = gr.Slider(minimum=-180, maximum=180, step=1, value=-160, label="Azimuth")

    with gr.Row():
        heat_map_origin = gr.Plot(label='处理前的热力图')
        heat_map_after = gr.Plot(label='处理后的热力图')
    # with gr.Row():
    #     scatter_origin = gr.Plot(label='处理前的QC的scatter')
    #     scatter_after = gr.Plot(label='处理后的QC的scatter')
    
    file_input.upload(preprocess, inputs=file_input, outputs=[pc_dist_sample_origin, pc_dist_QC_origin, pca_group_origin, pca_batch_origin, cv_mean_score_origin, heat_map_origin])
    # pre_process_button.click(preprocess, file_input, [pc_dist_sample_origin, pc_dist_QC_origin, pca_group_origin, pca_batch_origin, cv_mean_score_origin, heat_map_origin, scatter_origin])
    
    pca_batch_origin_ele.change(update_pca_origin_batch_plot, [pca_batch_origin_ele, pca_batch_origin_azi], pca_batch_origin)
    pca_batch_origin_azi.change(update_pca_origin_batch_plot, [pca_batch_origin_ele, pca_batch_origin_azi], pca_batch_origin)

    pca_group_origin_ele.change(update_pca_origin_group_plot, [pca_group_origin_ele, pca_group_origin_azi], pca_group_origin)
    pca_group_origin_azi.change(update_pca_origin_group_plot, [pca_group_origin_ele, pca_group_origin_azi], pca_group_origin)
    
    
    # with gr.Row():
    auc_compare = gr.Plot(label='处理前后的AUC对比')
    rsd = gr.Plot(label='处理前后的RSD对比')
    rsd_feature = gr.Plot(label='处理前后的RSD feature对比')

    pca_batch_after_ele.change(update_pca_after_batch_plot, [pca_batch_after_ele, pca_batch_after_azi], pca_batch_after)
    pca_batch_after_azi.change(update_pca_after_batch_plot, [pca_batch_after_ele, pca_batch_after_azi], pca_batch_after)

    pca_group_after_ele.change(update_pca_after_group_plot, [pca_group_after_ele, pca_group_after_azi], pca_group_after)
    pca_group_after_azi.change(update_pca_after_group_plot, [pca_group_after_ele, pca_group_after_azi], pca_group_after)

    
    process_button.click(process, method_radio, [pc_dist_sample_after, pc_dist_QC_after, pca_group_after, pca_batch_after, cv_mean_score_after, heat_map_after, auc_compare, rsd, rsd_feature, QC_correlation_origin, QC_correlation_after, output_file])



demo.launch(allowed_paths=["/"], share=False)
