import sys; sys.path.insert(0,'src')
from s4r.data.ingest import load_features
from s4r.features.coverage import coverage_confidence
df=load_features('data/processed/village_features.csv')
conf=coverage_confidence(df)
for i in range(len(df)):
    vid = int(df.iloc[i]['village_id'])
    c = conf[i]
    print(f'VID={vid:3d} conf={c:.4f}')
