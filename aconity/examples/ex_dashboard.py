"""
A quick example on how one might use the mongodb database to create a dashboard
for monitoring parameters while a job runs.

Before starting the dashboard, start AconitySTUDIO_process_data in another shell.

The dashboard can be seen in your Browser at `http://127.0.0.1:8050/`.
If no dashboard appears, look for a line like "Running on http://127.0.0.1:8050/"
during the startup of the script.
"""

#import things related to the dashboard
import dash
import dash_core_components as dcc
import dash_html_components as html
from collections import deque
import plotly.graph_objs as go

#std library
from collections import defaultdict
import sys
import time
import base64

#database access
from pymongo import MongoClient

#create app
app = dash.Dash('3dprinter-data')

#setup database
client = MongoClient()
db = client.dashboard_database
posts = db.posts

#get aconity logo
image_filename = 'logo-aconity3d.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
logo = html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))



def get_values(timeinterval = 60, topic=True, report=False):
    '''access database'''
    start = time.time() - timeinterval
    #basic_info = {'id':'MIDI', '_timestamp_db': {'$gt': start}}
    basic_info = {'_timestamp_db': {'$gt': start}}

    timestamps_topic = []
    values_topic = defaultdict(list)
    timestamps_report = []
    values_report = defaultdict(list)

    if topic:
        additional_info = {'topic':{'$exists':True}}
        search_for = {**additional_info, **basic_info} #merge dicts
        sensors_it = posts.find(search_for).sort('_timestamp_db') #returns iterator
        for data in sensors_it:
            #time (x-axis)
            timestamps_topic.append(data['_timestamp_db'])
            #all other values like temperature etc ... (y-axis)
            datalist = data['data']
            for item in datalist:
                cid = item['cid']
                #information is often duplicated
                sensordata = [dict(t) for t in {tuple(d.items()) for d in item['data']}] #remove duplicate dicts
                for measurement in sensordata:
                    name = '::' + measurement['name'] if (measurement['name'] != 'current_value') else ''
                    exact_component = cid + name
                    values_topic[exact_component].append(measurement['value'])
    if report:
        additional_info = {'report':{'$exists':True}}

    return {'times_topic': timestamps_topic, 'topic': values_topic,
            'times_report': timestamps_report, 'report': values_report}



#prepare the layout
options = list(get_values()['topic'].keys())
print('FULL LIST DROPDOWN CHOICES:')
for option in options:
    print('\t' + str(option))



#create the html page
app.layout = html.Div([
    html.Div([
        logo,
        html.H2('3D-Printer Topic Data',
                style={'float': 'right',
                       }),
        ]),
    dcc.Dropdown(id='3dprinter-data-name',
                 options=[{'label': s, 'value': s}
                          for s in options],
                 value = options[2:4],
                 multi=True
                 ),
    html.Div(children=html.Div(id='graphs'), className='row'),
    dcc.Interval(
        id='graph-update',
        interval=1000),
    ], className="container",style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000})


@app.callback(
    dash.dependencies.Output('graphs','children'),
    [dash.dependencies.Input('3dprinter-data-name', '59e9c5ad2800005200c9004b')],
    events=[dash.dependencies.Event('graph-update', '1000')]
    )
def update_graph(data_names):
    #only topic is supported. (not report)
    print('DATANAMES=%s' % data_names)
    graphs = []
    data = get_values()
    timestamps = data['times_topic']
    data_dict = data['topic']
    if len(data_names)>2:
        class_choice = 'col s12 m6 l4'
    elif len(data_names) == 2:
        class_choice = 'col s12 m6 l6'
    else:
        class_choice = 'col s12'


    for data_name in data_names:

        data = go.Scatter(
            x=timestamps,
            y=data_dict[data_name],
            name='Scatter',
            fill="tozeroy",
            fillcolor="#6897bb"
            )

        graphs.append(html.Div(dcc.Graph(
            id=data_name,
            animate=True,
            figure={'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(timestamps),max(timestamps)]),
                                                        yaxis=dict(range=[min(data_dict[data_name]),max(data_dict[data_name])]),
                                                        margin={'l':50,'r':1,'t':45,'b':1},
                                                        title='{}'.format(data_name))}
            ), className=class_choice))

    return graphs


#add external scripts. For example, they are responsible for allowing the graphs to reposition if one changes the size/shape of browser window.
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_css:
    app.scripts.append_script({'external_url': js})


if __name__ == '__main__':
    #fill in your ip and user data here
    login_data = {
        'rest_url' : f'http://143.167.195.109:9000',
        'ws_url' : f'ws://143.167.195.109:9000',
        'email' : 'admin@aconity3d.com',
        'password' : 'passwd'
    }

    app.run_server(debug=True)
