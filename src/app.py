## Importing libraries

## Basic modules
import numpy as np

## pandas
import pandas as pd

# from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def convertPSP(PSP_original, method):
    
    PSP_convert = PSP_original.copy()
    
    if method == 'abs2diff':
        # PSP-P-t#
        PSP_convert[PSP_convert.columns[0+1:5+1]] = (PSP_convert[PSP_convert.columns[0:5+1]].diff(axis=1))[PSP_convert.columns[0+1:5+1]]
        # PSP-P-P#
        PSP_convert[PSP_convert.columns[6+1:11+1]] = (PSP_convert[PSP_convert.columns[6:11+1]].diff(axis=1))[PSP_convert.columns[6+1:11+1]]
        # PSP-T-t#
        PSP_convert[PSP_convert.columns[12+1:14+1]] = (PSP_convert[PSP_convert.columns[12:14+1]].diff(axis=1))[PSP_convert.columns[12+1:14+1]]
        # PSP-T-T#
        PSP_convert[PSP_convert.columns[15+1:17+1]] = (PSP_convert[PSP_convert.columns[15:17+1]].diff(axis=1))[PSP_convert.columns[15+1:17+1]]
        
    elif method == 'diff2abs':
        # PSP-P-t#
        PSP_convert[PSP_convert.columns[0+1:5+1]] = (PSP_convert[PSP_convert.columns[0:5+1]].cumsum(axis=1))[PSP_convert.columns[0+1:5+1]]
        # PSP-P-P#
        PSP_convert[PSP_convert.columns[6+1:11+1]] = (PSP_convert[PSP_convert.columns[6:11+1]].cumsum(axis=1))[PSP_convert.columns[6+1:11+1]]
        # PSP-T-t#
        PSP_convert[PSP_convert.columns[12+1:14+1]] = (PSP_convert[PSP_convert.columns[12:14+1]].cumsum(axis=1))[PSP_convert.columns[12+1:14+1]]
        # PSP-T-T#
        PSP_convert[PSP_convert.columns[15+1:17+1]] = (PSP_convert[PSP_convert.columns[15:17+1]].cumsum(axis=1))[PSP_convert.columns[15+1:17+1]]

    elif method == 'diff2rDiff' or method == 'abs2rAbs':
        InputFeatures_toEliminate = ['PSP-P-P1', 'PSP-P-P6', 'PSP-T-T3']
        PSP_convert = PSP_convert.drop(InputFeatures_toEliminate, axis='columns')

    elif method == 'rDiff2diff':
        InputFeatures_toInsert = ['PSP-P-P1', 'PSP-P-P6', 'PSP-T-T3']
        InputFeatures_InsertPosition = [6, 11, 17]

        PSP_convert.insert(6, 'PSP-P-P1', pd.DataFrame(np.zeros(PSP_convert.shape[0])))
        PSP_convert.insert(11, 'PSP-P-P6', pd.DataFrame(5*np.ones(PSP_convert.shape[0])/1000 - PSP_convert[['PSP-P-P1', 'PSP-P-P2', 'PSP-P-P3', 'PSP-P-P4', 'PSP-P-P5']].sum(axis=1)))
        PSP_convert.insert(17, 'PSP-T-T3', -( 1-np.exp(-1) )*(PSP_original['PSP-T-T2']))

    elif method == 'rAbs2abs':
        print(method)
        InputFeatures_toInsert = ['PSP-P-P1', 'PSP-P-P6', 'PSP-T-T3']
        InputFeatures_InsertPosition = [6, 11, 17]

        PSP_convert.insert(6, 'PSP-P-P1', pd.DataFrame(np.zeros(PSP_convert.shape[0])))
        PSP_convert.insert(11, 'PSP-P-P6', pd.DataFrame(5*np.ones(PSP_convert.shape[0])/1000))
        PSP_convert.insert(17, 'PSP-T-T3', PSP_original['PSP-T-T2']-( 1-np.exp(-1) )*((PSP_original['PSP-T-T2']-PSP_original['PSP-T-T1'])))

    return PSP_convert    

def convertProcParam(ProcParam_original, method):

    # normalized ClntTemp = ClntTemp [degC] / 100 [degC]
    # normalized InjSpd = InjSpd [cm3/s] / 80 [cm3/s]
    # normalized PackPres = PackPres [bar] / 1000 [bar]
    # normalized PackTime = Packtime [s] / 60 [s]
    
    ProcParam_convert = ProcParam_original.copy()
    # ProcParam_convert = np.array(ProcParam_convert)

    if method == 'norm2abs':
        ProcParam_convert = ProcParam_convert*[100, 80, 1000, 60]

    if method == 'abs2norm':
        ProcParam_convert = ProcParam_convert/[100, 80, 1000, 60]

    return ProcParam_convert

def _dominates(trial0_objs, trial1_objs):
    # Non-dominated sorting defined in the NSGA-II paper
    # Non-dominated solution = Pareto optimial solution
    # trial0_objs = np.array(trial0_objs).tolist()[0]
    # trial1_objs = np.array(trial1_objs).tolist()[0]
    
    if trial0_objs == trial1_objs:
        return False
    
    return all(v0 <= v1 for v0, v1 in zip(trial0_objs, trial1_objs))

def _get_pareto_front_trials(Opt_Data, list_Objs):
    pareto_front_trials = []

    Data = Opt_Data[list_Objs].values.tolist() 
    # If the data is not converted to list, the speed is really×10000 slow

    for trial in range(len(Data)):
        dominated = False
        for other in range(len(Data)):
            if _dominates(Data[other], Data[trial]):
                dominated = True
                break

        if not dominated:
            pareto_front_trials.append(trial)

    return pareto_front_trials

def update_pareto_front_by_partialObjs(Opt_Data, Obj_list):
    # Get Pareto Front using partial objectives, and update
    
    new_Opt_Data = Opt_Data.copy()
    pareto_front_trials = _get_pareto_front_trials(new_Opt_Data, Obj_list)
    
    new_Optimal_status = []
    for trial in range(len(new_Opt_Data)):
        if trial in pareto_front_trials:
            new_Optimal_status.append(True)
        else:
            new_Optimal_status.append(False)
        
    new_Opt_Data['Optimal'] = new_Optimal_status
    
    return new_Opt_Data

PSP_ProcParam_Opt_Data = pd.read_csv('./Data/PSP_ProcParam_Opt_Data_TrialN5000.csv', index_col=0)
rDiffPSP = pd.DataFrame(data=[], columns=[
                                          'PSP-P-t1', 'PSP-P-t2', 'PSP-P-t3', 'PSP-P-t4', 'PSP-P-t5', 'PSP-P-t6',
                                          'PSP-P-P2', 'PSP-P-P3', 'PSP-P-P4', 'PSP-P-P5', 
                                          'PSP-T-t1', 'PSP-T-t2', 'PSP-T-t3', 
                                          'PSP-T-T1', 'PSP-T-T2'
])

PSP_ProcParam_Opt_Data['Temp_Symbol_size'] = PSP_ProcParam_Opt_Data['Obj_WMAPE']**-1

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

## Setting page layout ###################################################################################################################################

# explanation_text = 'This tool searches for optimized process parameters achieving a certain quality of the injection-molded product'\
#                     + ' based on in-mold condition centered approach using machine learning (ML) and explainable artificial intelligence (XAI).'\
#                     + ' More details will be found in the journal paper: Jinsu Gim, Chung-Yin Lin, and Lih-Sheng Turng, Paper Title, Journal Title (2023).'\
#                     + ' Publication record will be updated. Currently, the paper is been prepared. Any inquiry, send email to jgim@wisc.edu or turng@engr.wisc.edu.'


explanation_text = 'This tool searches for optimized process parameters achieving a certain quality of the injection-molded product'\
                    + ' based on in-mold condition centered approach using machine learning (ML) and explainable artificial intelligence (XAI).'\
                    + ' It is under development by Jinsu Gim, Chung-Yin Lin, and Lih-Sheng Turng, University of Wisconsin - Madison.'\
                    + ' Publications will be updated. Currently, the paper is under preparation. Any inquiries, please send an email to jgim@wisc.edu or turng@engr.wisc.edu.'

app.layout = html.Div(
                      dbc.Card(
                               dbc.CardBody([
                                            dbc.Row(dbc.Col(html.H4('PSP and ProcParam optimization')), align='center'), # Header
                                            dbc.Row(dbc.Col(html.Div(explanation_text))), html.Br(),
                                            # html.Br(), # Horizontal gap
                                            dbc.Row([
                                                     dbc.Col([
                                                              html.Div('Operations'),
                                                              html.Div('　1. Hover over a PSP optimization result in the 3D plot'),
                                                              html.Div('　2. Change the feasibility (WMAPE slider)'),
                                                              html.Div('　3. Change the accepatable quality range (Quality slider)'),
                                                     ]),
                                                     dbc.Col([
                                                              html.Div('　'),
                                                              html.Div('　4. Click a filtered PSP optimization result point in the 3D plot'),
                                                              html.Div('　5. Check the target and feasible PSP profiles in the line plot'),
                                                              html.Div('　6. Check the optimized Quality and ProcessParameters')
                                                     ])
                                            ]),
                                            # html.Br(), # Horizontal gap
                                            dbc.Row([
                                                     dbc.Col([html.Br(),
                                                              html.P('PSP optimization result for [Weight 37.0 g, Warp-ThickWall 0 mm, Warp-ThinWall 0 mm]'),
                                                              dbc.Col([
                                                                       dbc.Col(dcc.RadioItems([' Color by WMAPE　　', ' Color by Pareto Solution'], ' Color by Pareto Solution', id='Coloring',
                                                                                      labelStyle={'display': 'inline-block'})),
                                                              ]),
                                                              dcc.Graph(id='PSP_Opt_Graph')
                                                              ], width=8, style={}
                                                     ),
                                                     dbc.Col([html.Br(),html.Br(),
                                                              html.P('Filtering infeasible solutions'),
                                                              html.P('　Difference, WMAPE* [0-0.2]'),
                                                              dcc.RangeSlider(id='WMAPE_range-slider', 
                                                                              min=0, max=20/100, step=0.5/100, 
                                                                              marks={0:'0', 5/100:'0.05', 10/100:'0.1', 15/100:'0.15', 20/100:'0.2'}, 
                                                                              value=[0, 20/100]),
                                                              html.P('　*Weighted mean average percentage error'),
                                                              html.Br(),
                                                              html.P('Accepatable quality range'),
                                                              html.P('　PartWeight (g)'),
                                                              dcc.RangeSlider(id='PartWeight_range-slider', 
                                                                              min=36.0, max=38.5, step=0.1, 
                                                                              marks={36: '36', 36.5:'36.5', 37:'37', 37.5:'37.5', 38:'38', 38.5: '38.5'}, 
                                                                              value=[36.0, 38.5]),
                                                              html.P('　Warp-ThickWall (mm)'),
                                                              dcc.RangeSlider(id='Warp-ThickWall_range-slider', 
                                                                              min=-3.0, max=0, step=0.1, 
                                                                              marks={-3:'-3', -2.25:'-2.25', -1.5:'-1.5', -0.75:'-0.75', 0:'0'}, 
                                                                              value=[-3, 0]),
                                                              html.P('　Warp-ThinWall (mm)'),
                                                              dcc.RangeSlider(id='Warp-ThinWall-slider', 
                                                                              min=-2.5, max=1.5, step=0.1, 
                                                                              marks={-2.5:'-2.5', -1.5:'-1.5', -0.5:'-0.5', 0.5:'0.5', 1.5:'1.5'}, 
                                                                              value=[-2.5, 1.5]),
                                                     ], width=4),                                                             
                                            ]),
                                            # html.Br(), # Horizontal gap
                                            dbc.Row([
                                                     dbc.Col([html.Br(),
                                                              html.P('Comparison of Target and Feasible profiles'),
                                                              dcc.Graph(id='PSP_Profile_Graph')
                                                             ], 
                                                             width=8, style={}
                                                     ), # PSP profile plot
                                                     dbc.Col([html.Br(),
                                                              html.P('Predicted quality'),
                                                              html.Div(id='Table_Quality'),
                                                              html.Br(), html.Br(),
                                                              html.P('Optimized process parameter'),
                                                              html.Div(id='Table_ProcParam'),
                                                     ], width=4)
                                            ])
                               ])
                      )
)

##########################################################################################################################################################


## Update by range slider ################################################################################################################################

@app.callback(
              Output(component_id='PSP_Opt_Graph', component_property='figure'), 
              Input(component_id='WMAPE_range-slider', component_property='value'),
              Input(component_id='PartWeight_range-slider', component_property='value'),
              Input(component_id='Warp-ThickWall_range-slider', component_property='value'),
              Input(component_id='Warp-ThinWall-slider', component_property='value'),
              Input(component_id='Coloring', component_property='value')
)

def update_PSP_Opt_Chart_by_WMAPE(WMAPE_range, PartWeight_range, Warp_ThickWall_range, Warp_ThinWall_range, symbol_color_type):

    low_WMAPE, high_WMAPE = WMAPE_range
    low_PartWeight, high_PartWeight = PartWeight_range
    low_Warp_ThickWall, high_Warp_ThickWall = Warp_ThickWall_range
    low_Warp_ThinWall, high_Warp_ThinWall = Warp_ThinWall_range
    
    df = PSP_ProcParam_Opt_Data
    
    mask_WMAPE = (df['Obj_WMAPE'] >= low_WMAPE) & (df['Obj_WMAPE'] <= high_WMAPE)
    df_mask_newParetoFront = update_pareto_front_by_partialObjs(df[mask_WMAPE], ['Obj_PartWeight', 'Obj_Warp-ThickWall', 'Obj_Warp-ThinWall'])

    mask_PartWeight = (df_mask_newParetoFront['Pred_PartWeight (g)'] >= low_PartWeight) & (df_mask_newParetoFront['Pred_PartWeight (g)'] <= high_PartWeight)
    df_mask_newParetoFront = df_mask_newParetoFront[mask_PartWeight]
    
    mask_Warp_ThickWall = (df_mask_newParetoFront['Pred_Warp-ThickWall (mm)'] >= low_Warp_ThickWall) & (df_mask_newParetoFront['Pred_Warp-ThickWall (mm)'] <= high_Warp_ThickWall)
    df_mask_newParetoFront = df_mask_newParetoFront[mask_Warp_ThickWall]
    
    mask_Warp_ThinWall = (df_mask_newParetoFront['Pred_Warp-ThinWall (mm)'] >= low_Warp_ThinWall) & (df_mask_newParetoFront['Pred_Warp-ThinWall (mm)'] <= high_Warp_ThinWall)
    df_to_show = df_mask_newParetoFront[mask_Warp_ThinWall]
    
    fig = px.scatter_3d(
                        df_to_show,
                        x='Obj_PartWeight', y='Obj_Warp-ThickWall', z='Obj_Warp-ThinWall',
                        color='Obj_WMAPE' if symbol_color_type == ' Color by WMAPE　　' else 'Optimal',
                        color_continuous_scale='Plotly3',
                        symbol='Optimal', 
                        size='Temp_Symbol_size', size_max=20, 
                        opacity=1.0,
                        range_color=(0, 20/100),
                        hover_data={'Obj_PartWeight':False,
                                    'Obj_Warp-ThickWall':False,
                                    'Obj_Warp-ThinWall':False,
                                    'Obj_WMAPE':False,
                                    'Temp_Symbol_size':False,
                                   }
                        )

    fig.update_traces(customdata=df_to_show['Trial#'])

    fig.update_layout(
        scene=dict(
            xaxis = dict(title='Obj: PartWeight', range=[0, max(df_mask_newParetoFront['Obj_PartWeight'])*1.1]),
            yaxis = dict(title='Obj: Warp-ThickWall', range=[0, max(df_mask_newParetoFront['Obj_Warp-ThickWall'])*1.1]),
            zaxis = dict(title='Obj: Warp-ThinWall', range=[0, max(df_mask_newParetoFront['Obj_Warp-ThinWall'])*1.1])
        ),
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.932),
        scene_aspectmode='cube',
        legend_title_text='Optimal?',
        showlegend=True,
        coloraxis_colorbar_x=-0.15,
        coloraxis_colorbar_title_text='WMAPE',
        margin=dict(l=0, r=0, t=0, b=0),
        # height=550, width=700
    )
    
    return fig
##########################################################################################################################################################


## Update by click data ##################################################################################################################################

@app.callback(
              [Output(component_id='PSP_Profile_Graph', component_property='figure'),
              Output(component_id='Table_Quality', component_property='children'),
              Output(component_id='Table_ProcParam', component_property='children')],
              [Input(component_id='PSP_Opt_Graph', component_property='clickData')]
)
def display_click_PSP(click_data):
    
    if click_data == None:
        Trial_num = 0
        opacity=0
        
    else:
        Trial_num = click_data['points'][0]['customdata']
        opacity=1
        
    click_OptData = PSP_ProcParam_Opt_Data[ PSP_ProcParam_Opt_Data['Trial#'] == Trial_num ]

    ## PSP profile display
    rDiffPSP_target_toPlot = click_OptData[['Sugg-rDiffPSP-P-P2', 'Sugg-rDiffPSP-P-P3', 'Sugg-rDiffPSP-P-P4','Sugg-rDiffPSP-P-P5', 
                                       'Sugg-rDiffPSP-P-t1', 'Sugg-rDiffPSP-P-t2', 'Sugg-rDiffPSP-P-t3', 'Sugg-rDiffPSP-P-t4', 'Sugg-rDiffPSP-P-t5', 'Sugg-rDiffPSP-P-t6', 
                                       'Sugg-rDiffPSP-T-T1', 'Sugg-rDiffPSP-T-T2', 'Sugg-rDiffPSP-T-t1', 'Sugg-rDiffPSP-T-t2','Sugg-rDiffPSP-T-t3']].copy()
    
    for column_name in rDiffPSP_target_toPlot.columns:
        rDiffPSP_target_toPlot.rename(columns={column_name:column_name[10:]}, inplace=True)
    absPSP_target_toPlot = convertPSP(convertPSP(rDiffPSP_target_toPlot[rDiffPSP.columns], 'rDiff2diff'), 'diff2abs')

    diffPSP_optimized_toPlot = click_OptData[['Feas-DiffPSP-P-t1', 'Feas-DiffPSP-P-t2', 'Feas-DiffPSP-P-t3', 'Feas-DiffPSP-P-t4', 'Feas-DiffPSP-P-t5', 'Feas-DiffPSP-P-t6',
                                          'Feas-DiffPSP-P-P1', 'Feas-DiffPSP-P-P2', 'Feas-DiffPSP-P-P3', 'Feas-DiffPSP-P-P4', 'Feas-DiffPSP-P-P5', 'Feas-DiffPSP-P-P6',
                                          'Feas-DiffPSP-T-t1', 'Feas-DiffPSP-T-t2', 'Feas-DiffPSP-T-t3', 'Feas-DiffPSP-T-T1', 'Feas-DiffPSP-T-T2','Feas-DiffPSP-T-T3']].copy()
    
    for column_name in diffPSP_optimized_toPlot.columns:
        diffPSP_optimized_toPlot.rename(columns={column_name:column_name[9:]}, inplace=True)
    absPSP_optimized_toPlot = convertPSP(diffPSP_optimized_toPlot, 'diff2abs')

    Time_P_target = np.array(absPSP_target_toPlot).flatten()[0:6]*60
    Time_P_target = np.insert(Time_P_target, 0, 0)
    Time_P_target = np.append(Time_P_target, 60)

    Value_P_target = np.array(absPSP_target_toPlot).flatten()[6:12]*1000
    Value_P_target = np.insert(Value_P_target, 0, 0)
    Value_P_target = np.append(Value_P_target, 0)

    Time_T_target = np.array(absPSP_target_toPlot).flatten()[12:15]*60
    Time_T_target = np.insert(Time_T_target, 0, 0)
    Time_T_target = np.append(Time_T_target, 60)

    Value_T_target = np.array(absPSP_target_toPlot).flatten()[15:18]*205
    Value_T_target = np.insert(Value_T_target, 0, Value_T_target[0])
    Value_T_target = np.append(Value_T_target, Value_T_target[0])
    
    Time_P_optimized = np.array(absPSP_optimized_toPlot).flatten()[0:6]*60
    Time_P_optimized = np.insert(Time_P_optimized, 0, 0)
    Time_P_optimized = np.append(Time_P_optimized, 60)
    
    Value_P_optimized = np.array(absPSP_optimized_toPlot).flatten()[6:12]*1000
    Value_P_optimized = np.insert(Value_P_optimized, 0, 0)
    Value_P_optimized = np.append(Value_P_optimized, 0)

    Time_T_optimized = np.array(absPSP_optimized_toPlot).flatten()[12:15]*60
    Time_T_optimized = np.insert(Time_T_optimized, 0, 0)
    Time_T_optimized = np.append(Time_T_optimized, 60)

    Value_T_optimized = np.array(absPSP_optimized_toPlot).flatten()[15:18]*205
    Value_T_optimized = np.insert(Value_T_optimized, 0, Value_T_optimized[0])
    Value_T_optimized = np.append(Value_T_optimized, Value_T_optimized[0])

    P_target = pd.DataFrame(data = [Time_P_target, Value_P_target], index=['Time (s)', 'Cavity pressure (bar)']).T
    T_target = pd.DataFrame(data = [Time_T_target, Value_T_target], index=['Time (s)', 'Mold surface temperature (degC)']).T
    P_optimized = pd.DataFrame(data = [Time_P_optimized, Value_P_optimized], index=['Time (s)', 'Cavity pressure (bar)']).T
    T_optimized = pd.DataFrame(data = [Time_T_optimized, Value_T_optimized], index=['Time (s)', 'Mold surface temperature (degC)']).T    

    fig_PSP_plot = make_subplots(specs=[[{"secondary_y": True}]])
    fig_PSP_plot.add_trace(go.Scatter(
                                      x=P_target['Time (s)'], y=P_target['Cavity pressure (bar)'], opacity=opacity,
                                      name='Target P',
                                      line=dict(color='royalblue', width=4)
                                      ),
                            secondary_y=False
                          )
    fig_PSP_plot.add_trace(go.Scatter(
                                      x=T_target['Time (s)'], y=T_target['Mold surface temperature (degC)'], opacity=opacity,
                                      name='Target T',
                                      line=dict(color='orange', width=4)
                                     ),
                            secondary_y=True
                          )
    fig_PSP_plot.add_trace(go.Scatter(
                                      x=P_optimized['Time (s)'], y=P_optimized['Cavity pressure (bar)'], opacity=opacity,
                                      name='Feasible P',
                                      line=dict(color='royalblue', width=4, dash='dot')
                                      ),
                            secondary_y=False
                          )
    fig_PSP_plot.add_trace(go.Scatter(
                                      x=T_optimized['Time (s)'], y=T_optimized['Mold surface temperature (degC)'], opacity=opacity,
                                      name='Feasible T',
                                      line=dict(color='orange', width=4, dash='dot')
                                      ),
                            secondary_y=True
                          )

    fig_PSP_plot.update_layout(
                               xaxis_range=[0, 14],
                               legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.932),
                               height=350,
                               margin=dict(l=0, r=0, t=0, b=0),
    )
    fig_PSP_plot.update_xaxes(title_text="Time (s)")
    fig_PSP_plot.update_yaxes(range = [-20, 600], title_text="Cavity pressure (bar)", secondary_y=False)
    fig_PSP_plot.update_yaxes(range = [18, 80], title_text="Mold surface temperature (℃)", secondary_y=True)

    ## Optimization result table display
    if click_data == None:
        Quality_to_show = pd.DataFrame(data=['-', '-', '-'], index=['PartWeight (g)', 'Warp-ThickWall (mm)', 'Warp-ThinWall (mm)']).T
        ProcParam_to_show = pd.DataFrame(data=['-', '-', '-', '-'], index=['ClntTemp (℃)', 'InjSpd (cm³/s)', 'PackPres (bar)', 'PackTime (s)']).T
    else:
        Quality_to_show = click_OptData[['Pred_PartWeight (g)', 'Pred_Warp-ThickWall (mm)', 'Pred_Warp-ThinWall (mm)']].copy()
        for column_name in Quality_to_show.columns:
            Quality_to_show.rename(columns={column_name:column_name[5:]}, inplace=True)
            
        ProcParam_to_show = click_OptData[['Opt_ClntTemp', 'Opt_InjSpd', 'Opt_PackPres', 'Opt_PackTime']].copy()*[100, 80, 1000, 60]
        ProcParam_to_show.columns=['ClntTemp (℃)', 'InjSpd (mm³/s)', 'PackPres (bar)', 'PackTime (s)']

    tbl_Quality = dbc.Table.from_dataframe(Quality_to_show.round(2))
    tbl_ProcParam = dbc.Table.from_dataframe(ProcParam_to_show.round(2))
    
    return fig_PSP_plot, tbl_Quality, tbl_ProcParam
    
##########################################################################################################################################################

app.run_server(host = "localhost")