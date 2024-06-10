from dataclasses import dataclass
from itertools import zip_longest
from PIL import ImageColor
from benchtool.Analysis import overall_solved, task_average
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
from typing import Literal, Optional
import seaborn as sns
import matplotlib.pyplot as plt
import bisect


pd.options.mode.chained_assignment = None  # default='warn'


def light_gradient(rgb: tuple[int, int, int], n: int) -> list[tuple[int, int, int]]:
    top: tuple[int, int, int] = (240, 241, 241)

    gradient = list(
        map(lambda x: 'rgb' + str(tuple(x)),
            np.linspace(rgb, top, n).astype(int).tolist()))

    return gradient


@dataclass
class Bar:
    name: str
    values: list[int]
    color: str = None

import code

def stacked_barchart_times(
    case: str,
    df: pd.DataFrame,
    limits: list[float],
    limit_type: str,
    strategies: list[str] = None,
    colors: list[str] = None,
    show: bool = True,
    image_path: Optional[str] = None,
    agg: Literal['any', 'all'] = 'all',
    manual_bars: list[Bar] = [],
):

    df = df[df['workload'] == case]

    if not strategies:
        strategies = df.strategy.unique()

    tasks = df.task.unique()
    total_tasks = len(tasks)

    results = pd.DataFrame(columns=limits, index=strategies, dtype=int, data=0)

    results['rest'] = total_tasks

    for within in limits:
        dft = overall_solved(df, agg=agg, within=within, solved_type=limit_type)
        dft = dft.reset_index()
        dft = dft.groupby(['strategy']).sum(numeric_only=False)
        for strategy in strategies:
            # Note: I think the new version of Pandas broke some of this code.
            # Use 1.5.3 for now and come back and fix.
            results.loc[strategy].loc[within] = dft.loc[strategy]['solved'] - (
                total_tasks - results.loc[strategy].loc['rest'])
            results.loc[strategy].loc[
                'rest'] = results.loc[strategy].loc['rest'] - results.loc[strategy].loc[within]

    results = results.rename_axis('strategy')
    results = results.reset_index()

    results = results.melt(id_vars=['strategy'], value_vars=limits + ['rest'])
    grouped_df = results.groupby('strategy')

    strategy_to_times = {}
    for (strategy, df_strat) in df.groupby('strategy'):
        if strategy not in strategies:
            continue
        times = []
        for (_, df_task) in df_strat.groupby('task'):
            import code
            num_trials = len(df_task)
            assert num_trials == 5 # remove if we add short circuiting
            assert num_trials % 2, "num trials must be odd"
            majority_found = df_task.foundbug.sum() * 2 > num_trials
            # if df_task.foundbug.sum() not in [0, 5]:
            #     code.interact(local=locals())
            # if df_task.foundbug.all():
            if majority_found:
                # use nsmallest(num_tasks // 2 + 1) like so if we have short circuiting:
                # pd.Series([50, 40, 10, 30]).nsmallest(3).iloc[-1]
                t = df_task.time.median()
                times.append(t)
                assert t != 60
        times.sort()
        strategy_to_times[strategy] = times
    trivial_cutoff_time = 0.5
    num_found_by_all_in_cutoff = min(
        bisect.bisect_left(times, trivial_cutoff_time)
        for times in strategy_to_times.values()
    )
    max_found = max(len(times) for tiems in strategy_to_times.values())
    strat_names = {
        "TypeBasedGenerator": "QuickChick type-based generator",
        "LGenerator": "Our type-based generator (OTBG)",
        "LEqGenerator": "OTBG tuned for diverse (w.r.t default equality) valid BSTs",
        "LExceptGenerator": "OTBG tuned for diverse (w.r.t. BST structure) valid BSTs",
    }
    sns.set_context("paper")
    sns.set_style("whitegrid",{'font.family':'serif', 'font.serif':'Times New Roman', 'axes.edgecolor': 'white','grid.color': '#DCDCDC'})
    # https://seaborn.pydata.org/tutorial/color_palettes.html#using-categorical-color-brewer-palettes

    palette = sns.color_palette("tab10")
    reordered = [palette[i] for i in [0, 2, 1, 3]]
    sns.set_palette(reordered, 4)

    # set2 = [
    #     (0.9882352941176471, 0.5529411764705883, 0.3843137254901961), # 2
    #     (0.4, 0.7607843137254902, 0.6470588235294118), # 1
    #     (0.5529411764705883, 0.6274509803921569, 0.796078431372549), # 3
    #     (0.9058823529411765, 0.5411764705882353, 0.7647058823529411), # 4
    #     (0.6509803921568628, 0.8470588235294118, 0.32941176470588235), # 5
    #     (1.0, 0.8509803921568627, 0.1843137254901961), # 6
    #     (0.8980392156862745, 0.7686274509803922, 0.5803921568627451), # 7
    #     (0.7019607843137254, 0.7019607843137254, 0.7019607843137254) # 8
    # ]
    # sns.set_palette(set2, 4)
    # for strategy, times in strategy_to_times.items():
    #     x_key = "Time" 
    #     y_key = "# bugs found"
    #     ax = sns.lineplot(
    #         data={
    #             x_key : [0] + times + [60],
    #             y_key : list(range(len(times) + 2)),
    #         },
    #         x=x_key,
    #         y=y_key,
    #         label=strat_names.get(strategy, strategy),
    #         alpha=0.7,
    #     )
    #     plt.setp(ax.lines, alpha=.3)
    #     plt.setp(ax.collections, alpha=.3)
    
    # via a dataframe
    plt.clf()
    data = []
    for strategy, times in sorted(
        strategy_to_times.items(),
        key=lambda p: list(strat_names.keys()).index(p[0])
    ):
        time_points = [0] + times + [60]
        bugs_found = list(range(len(times) + 2))
        for t, b in zip(time_points, bugs_found):
            data.append({
                "Time (s)": t,
                "# bugs found": b,
                "Strategy": strat_names.get(strategy, strategy),
            })
    df = pd.DataFrame(data)
    sns.lineplot(
        data=df,
        x="Time (s)",
        y="# bugs found",
        hue="Strategy",
        style="Strategy",
        # alpha=0.7,
        linewidth=4,
    )

    plt.gca().set_ylim(bottom=num_found_by_all_in_cutoff,top=max_found + 2)
    plt.legend(title='Strategy', loc='lower right')
    plt.savefig(f"{image_path}/{case}-med-times.svg")

    with open(f"{image_path}/{case}-med-times.csv", "w") as f:
        tab = '\t'
        f.write(tab.join(strategy for strategy in strategies))
        f.write('\n')
        for row in zip_longest(*[strategy_to_times[strategy] for strategy in strategies], fillvalue=''):
            f.write(tab.join(map(str, row)))
            f.write('\n')
            # f.write(f"{strategy}\t{tab.join(map(str,times))}\n")


    cts_keys = [(list(item.value), key) for key, item in grouped_df]
    cts_keys.sort(reverse=True)
    with open(f"{image_path}/{case}.csv", "w") as f:
        f.writelines(
            f"{key}\t" + "\t".join(map(str, cts)) + "\n"
            for cts, key in cts_keys
        )


    if not colors:
        colors = [
            '#000000',  # black
            '#900D0D',  # red
            '#DC5F00',  # orange
            '#243763',  # blue
            '#436E4F',  # green
            '#470938',  # purple
            '#D61C4E',  # pink
            '#334756',  # dark blue
            '#290001',  # dark brown
            '#000000',  # black
        ]

    extrapolated_colors = list(
        map(light_gradient, map(ImageColor.getrgb, colors), [len(limits) + 1] * len(colors)))

    fig = go.Figure()
    fig.update_layout(
        title=f'',
        xaxis=go.layout.XAxis(showticklabels=False,),
        yaxis=go.layout.YAxis(
            title='',
            showticklabels=True,
        ),
        font_size=40,
        font={'family': 'Helvetica'},
        width=6000,
        height=500,
        showlegend=False,
    )
    fig.update_layout(
        yaxis={
            "tickfont": {"size": 40},
        }
    )

    # hide y axis title

    strategy_sorter = dict(map(lambda x: (x[1], x[0]), enumerate(strategies)))

    strategies = sorted(strategies,
                        key=lambda x: strategy_sorter[x] if x in strategy_sorter.keys() else -1)

    def no_gen(s):
        s = ''.join(s.split('Generator'))
        # if s == "ManualTypeBased2":
        #     return "OurTypeBased"
        # if s == "CondEntropyEnvariants":
        #     return "Conditional Entropy + Invariants"
        # elif s == "ManualTypeBased2":
        #     return "OurTypeBased"
        return s
    for strategy, color in zip(strategies[::-1], extrapolated_colors):
        fig.add_trace(
            go.Bar(
                x=results[results['strategy'] == strategy]['value'],
                y=list(map(no_gen, results[results['strategy'] == strategy]['strategy'])),
                name=''.join(strategy.split('G')),
                marker_color=color,
                text=results[results['strategy'] == strategy]['value'],
                orientation='h',
                width=0.8,
                textposition='auto',
                textfont_size=60,
                textfont={'family': 'Helvetica'},
                textangle=0,
                cliponaxis=False,
                insidetextanchor='middle',
                # don't show name on y axis
            ))

    for bar in manual_bars:
        fig.add_trace(
            go.Bar(
                x=bar.values,
                y=np.array([bar.name] * (len(limits) + 1)),
                name=bar.name,
                marker_color=light_gradient(ImageColor.getrgb(bar.color),
                                            len(limits) + 1),
                text=bar.values,
                orientation='h',
                width=0.8,
                textposition='auto',
                textfont_size=60,
                textfont={'family': 'Helvetica'},
                textangle=0,
                cliponaxis=False,
                insidetextanchor='middle',
            ))

    if image_path:
        fig.write_image(f'{image_path}/{case}.png',
                        width=1600,
                        height=900,
                        scale=1,
                        engine='kaleido')

    if show:
        fig.show()


def dashboard(df: pd.DataFrame):
    app = Dash(__name__)

    div_style = {'width': '31%', 'float': 'left', 'display': 'inline-block', 'margin-right': '15px'}
    app.layout = html.Div([
        html.Div([
            html.Div([dcc.Dropdown(sorted(df['workload'].unique()), 'BST', id='workload')],
                     style=div_style),
            html.Div([
                dcc.Dropdown(['time', 'inputs'], 'time', id='col'),
                dcc.RadioItems(['linear', 'log'], 'linear', id='yscale', inline=True)
            ],
                     style=div_style),
            html.Div([
                dcc.Dropdown(
                    df['strategy'].unique(), df['strategy'].unique(), id='strategies', multi=True)
            ],
                     style={
                         'width': '31%',
                         'display': 'inline-block'
                     })
        ],
                 style={'margin-bottom': '15px'}),
        dcc.Graph(id='graph', style={'height': '85vh'})
    ])

    @app.callback(
        Output('graph', 'figure'),
        Input('workload', 'value'),
        Input('col', 'value'),
        Input('yscale', 'value'),
        Input('strategies', 'value'),
    )
    def update_graph(workload, col, yscale, strategies):
        dff = df[(df['workload'] == workload) & (df['strategy'].isin(strategies))]

        # Note: this only includes tasks that everyone solved
        dff = task_average(dff, col)

        dff = dff.reset_index()
        fig = px.bar(dff,
                     x='task',
                     y=col,
                     color='strategy',
                     barmode='group',
                     error_y=col + '_std',
                     log_y=yscale == 'log')
        return fig

    return app