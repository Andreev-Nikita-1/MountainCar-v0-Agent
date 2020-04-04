import numpy as np
import plotly.graph_objects as go
import plotly.offline
from plotly.subplots import make_subplots


def learning_visualization(games, rewards):
    if len(games) <= 100:
        data_ = games
    else:
        data_ = np.array(games)[[i % (len(games) // 100) == 0 for i in range(len(games))]]
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[1],
        subplot_titles=("trajectory", "rewards"))
    for game, text, num in data_:
        xs, ys, xs1, ys1, acs = [], [], [], [], []
        for s, a in game:
            xs.append(s[0])
            ys.append(s[1])
            if a != 1:
                xs1.append(s[0])
                ys1.append(s[1])
                acs.append(int(a))

        fig.add_trace(go.Scatter(
            visible=False,
            x=xs1, y=ys1, mode='markers',
            name="actions",
            marker=dict(color=acs)
        ),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            visible=False,
            x=xs, y=ys, mode='lines',
            name="trajectory"
        ),
            row=1, col=1)
    fig.add_trace(go.Scatter(
        visible=True,
        x=[i + 1 for i in range(len(rewards))], y=rewards, mode='lines',
        name="rewards",
    ),
        row=1, col=2)
    fig.data[0].visible = True
    fig.data[1].visible = True
    steps = []
    for i, tuple in enumerate(data_):
        _, text, _ = tuple
        step = dict(
            method="update",
            args=[{"visible": [t in [2 * i, 2 * i + 1, len(fig.data) - 1] for t in range(len(fig.data))]},
                  {'title.text': text}],
        )
        steps.append(step)
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "game: "},
        pad={"t": len(fig.data) // 2},
        steps=steps
    )]

    fig.update_layout(
        xaxis_title="state[0]",
        yaxis_title="state[1]",
        sliders=sliders,
        title={'text': 'trajectory'},
        yaxis=dict(range=[-0.07, 0.07]),
        xaxis=dict(range=[-1.2, 0.6])
    )

    fig.update_yaxes(title_text="state[1]", range=[-0.07, 0.07], row=1, col=1)
    fig.update_xaxes(title_text="state[0]", range=[-1.2, 0.6], row=1, col=1)
    fig.update_yaxes(title_text="average reward", row=1, col=2)
    fig.update_xaxes(title_text="epoch", row=1, col=2)

    plotly.offline.plot(fig, filename='learning.html')
    fig.show()
