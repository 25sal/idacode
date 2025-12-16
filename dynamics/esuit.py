import pandas as pd
import glob
import logging
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


data_folder = 'data/CulturalDeepfake/'
results = {}


views = 0
total_mintime = None
total_maxtime = None
spread_times = []
authors = set()
total_events = None

files = glob.glob(data_folder + 'posts/*.csv')
logger.info(f'Found {len(files)} files to process.')

timeseries = {}
for file in files:
    basename = os.path.basename(file)
    results[basename] = {}
    logger.info(f'Processing file: {file}')
    df = pd.read_csv(file)
    
    views += df.iloc[0]['ReactionsCount'] + df.iloc[0]['SubCommentsCount']
    df['datetime'] = pd.to_datetime(df['CommentAt'], format='%m/%d/%Y %H:%M:%S %p')
    
    if total_events is None:
        total_events = df['datetime']
    else:
        total_events = pd.concat([total_events, df['datetime']])

    mintime = df['datetime'].min()
    maxtime = df['datetime'].max()
    
    if total_mintime is None:
        total_mintime = mintime
    else:
        total_mintime = min(total_mintime, mintime)
    if total_maxtime is None:
        total_maxtime = maxtime
    else:
        total_maxtime = max(total_maxtime, maxtime)
   

    df['spread_time'] = df['datetime'] - mintime
    sorted = df['spread_time'].sort_values()
    min_spread_time = sorted.iloc[1]
    
    #convert nanoseconds to minutes
    min_spread_time = min_spread_time / pd.Timedelta(minutes=1)
    max_spread_time = sorted.iloc[-1]
    max_spread_time = max_spread_time / pd.Timedelta(minutes=1)
    spread_times.append(min_spread_time)
    
    # collect unique authors
    authors.update(df['Author'].unique())
    
    # group events by datetime column and count them
    logger.info('Grouping events')
    total_events = pd.DataFrame(df, columns=['datetime'])
    total_events = total_events.dropna()
    total_events = total_events.sort_values(by='datetime')
    total_events = total_events.reset_index(drop=True)


    # count events in interval
    total_events = total_events.groupby(total_events['datetime'].dt.floor('h')).count()
    timeseries[basename] = total_events

    plt.figure()
    ax = total_events.plot(title=f'Dynamic of post {basename}', kind='bar', figsize=(10, 8))
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of comments')
    ax.set_xticks(ax.get_xticks()[::2])  # una label ogni 2 ore
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))  # max 8 tick
    plt.savefig(f"{data_folder}output/{os.path.basename(file)[:-3]}_events_per_hour.png")
    results[basename] = {
        'min_spread_time_minutes': min_spread_time,
        'max_spread_time_minutes': max_spread_time,
        'views': int(df.iloc[0]['ReactionsCount']+df.iloc[0]['SubCommentsCount']),
        'unique_authors': len(df['Author'].unique()),
        'CLS_days': (maxtime - mintime) / pd.Timedelta(days=1),
        'GR': (df.iloc[0]['ReactionsCount']+df.iloc[0]['SubCommentsCount']) / ((maxtime - mintime) / pd.Timedelta(days=1))   
    }
    # count views
    views = views + df.iloc[0]['ReactionsCount']+df.iloc[0]['SubCommentsCount']
    
plt.figure()
for key, ts in timeseries.items():
    print(ts.head())
    t = ts.index - ts.index[0]
    plt.plot(t,ts/ts.max(), label=key)
plt.savefig(f"{data_folder}output/aggregated_events_per_hour.png")
logger.info(f'VIEWS: {views}')
#content life span
CLS = total_maxtime-total_mintime
# convert to days
CLS = CLS / pd.Timedelta(days=1)
logger.info(f'Total content life span (days): {CLS} days')
#Growth Rate
GR = views / CLS
logger.info(f'GR: {GR}')
logger.info(f'MINSPREADTIME: {min_spread_time}')
# count uniques values in 'Author' column
UR = len(authors)
logger.info(f'Number of unique authors: {UR}')
results['aggregated'] = {
    'views': str(views),
    'CLS_days': str(CLS),
    'GR': str(GR),
    'MINSPREADTIME': str(min_spread_time),
    'UR': str(UR)
}

# save results to json
import json
with open(f'{data_folder}output/summary_results.json', 'w') as f:
    json.dump(results, f, indent=4)
