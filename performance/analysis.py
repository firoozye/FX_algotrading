import plotly.express as px
import pandas as pd
import plotly.express as px
import utils.settings as settings
# import kaleido

def plot_lines(dataframe, columns=None, value_names='price',
               column_names = 'contract', filename_prefix=None):

    if columns is not None and (len(columns) > 0):
        plot_data = dataframe[columns]
    else:
        plot_data = dataframe.copy()

    if filename_prefix is None:
        filename_prefix =''

    plot_data.index.name = 'date'
    plot_data.index = pd.to_datetime(plot_data.index)
    if plot_data.shape[1] > 1:
        plot_data = plot_data.stack()
        plot_data = pd.DataFrame(plot_data)  # it was a series
        plot_data.columns = [value_names]  # only one col left!
        plot_data = plot_data.reset_index()
        plot_data = plot_data.rename(columns={'level_1': column_names})
        # Plot only four lines at a time
        contracts = plot_data[column_names].unique()
        step = 1
        for ind in range(0, len(contracts), step):
            sets = contracts[ind:ind + step]
            subdf = plot_data.loc[plot_data[column_names].map(lambda x: x in sets), :]
            fig = px.line(subdf, x='date', y=value_names, color=column_names)
            ctc = contracts[ind]
            # cannot write to settings.OUTPUT_FIles. Why?
            fig.write_image(settings.OUTPUT_FIGS + filename_prefix + f'{ctc}.png', format='png') #, engine=kaleido)
            print(f'Wrote {contracts[ind:ind+step]}')
    else:
        if value_names is None:
            value_names = 'perf'

        plot_data.columns = [value_names]
        plot_data = plot_data.reset_index()
        fig = px.line(plot_data, x='date', y=value_names)
        # camnnot write to settings.OUTPUT_FILES
        fig.write_image(settings.OUTPUT_FIGS + filename_prefix + f'{value_names}.png', format='png')
        print(f'wrote plot {filename_prefix}{value_names}.png')
    return

# df = px.data.gapminder().query("country=='Canada'")
# fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
# fig.show()
#
#
#
# df = px.data.stocks()
# fig = px.line(df, x='date', y="GOOG")
# fig.show()
#
#
# fig.write_image("fig1.pdf")
#
# rolled_data = pd.read_parquet('~/Dropbox/CCA/all_continuous_stir_futures.pqt')
# rolled_data.columns = rolled_data.columns.swaplevel(0,1)
# prices = rolled_data.loc[:,'ADJ_PX_MID']
# prices.index = pd.to_datetime(prices.index)
# prices.index.name = 'date'
# prices = prices.reset_index()
# non_date = [x for x in prices.columns if x != 'date']
# for ind in range(0,len(non_date),4):
#     sets = non_date[ind:ind+4]
#     fig = px.line(prices, x='date', y=sets)
#     ctc = non_date[ind]
#     fig.write_image(f'{ctc}.pdf')
#
# prices = prices.set_index('date')
#
# prices = prices.set_index('date')
# prices = prices.unstack()
# prices.name = 'prices'
# prices = pd.DataFrame(prices)
#
# prices = prices.reset_index()
# prices = prices.rename(columns={'level_0':'contract'})
#
# contracts = prices['contract'].unique()
# for ind in range(0,len(contracts),4):
#     sets = contracts[ind:ind+4]
#     subdf = prices.loc[prices['contract'].map(lambda x: x in sets),:]
#     fig = px.line(subdf,x='date', y='prices',color='contract')
#     ctc = contracts[ind]
#     fig.write_image(f'{ctc}.pdf')
#
#
#
#
# import plotly.graph_objects as go
# fig = go.Figure(
#     data=[go.Bar(y=[2, 1, 3])],
#     layout_title_text="A Figure Displayed with fig.show()"
# )
# fig.show()
#
#
#
#
# from dash import Dash, dcc, html, Input, Output
# import plotly.express as px
#
# app = Dash(__name__)
#
#
# app.layout = html.Div([
#     html.H4('Life expectancy progression of countries per continents'),
#     dcc.Graph(id="line-charts-x-graph"),
#     dcc.Checklist(
#         id="line-charts-x-checklist",
#         options=["Asia", "Europe", "Africa","Americas","Oceania"],
#         value=["Americas", "Oceania"],
#         inline=True
#     ),
# ])
#
#
# @app.callback(
#     Output("line-charts-x-graph", "figure"),
#     Input("line-charts-x-checklist", "value"))
# def update_line_chart(continents):
#     df = px.data.gapminder() # replace with your own data source
#     mask = df.continent.isin(continents)
#     fig = px.line(df[mask],
#         x="year", y="lifeExp", color='country')
#     return fig
#

if __name__ == "__main__":
    app.run_server(port=2223, debug=True)