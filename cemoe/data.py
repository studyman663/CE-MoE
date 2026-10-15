"""Named-column input, explicit target semantics, and train-only preprocessing."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

MAGNETIC = [f'{i}.{axis} (nT)' for i in range(3) for axis in 'XYZ']
FEATURES = MAGNETIC + ['inclination_angle (°)', 'azimuth_angle (°)', 'temperature (°)']
TARGET = 'distance (m)'


def read_frame(path, target_mode='signed', require_target=True):
    path = Path(path)
    frame = pd.read_excel(path) if path.suffix.lower() == '.xlsx' else pd.read_csv(path)
    required = FEATURES + ([TARGET] if require_target else [])
    if not set(required).issubset(frame.columns):
        raise ValueError(f'Missing columns: {set(required) - set(frame.columns)}')
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError('Inputs/labels must be numeric and finite; no silent imputation.')
    if require_target:
        if target_mode == 'absolute':
            frame[TARGET] = frame[TARGET].abs()
        elif target_mode == 'nonnegative':
            if (frame[TARGET] < 0).any():
                raise ValueError('Negative labels: explicitly choose signed or absolute semantics.')
        elif target_mode != 'signed':
            raise ValueError('target_mode must be signed, absolute, or nonnegative.')
    return frame


def metrics(y, prediction, target_mode='nonnegative'):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if not len(y) or y.shape != prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Metrics require nonempty, matching, finite vectors.')
    err = np.abs(y-prediction)
    result = dict(n=len(y), rmse=float(np.sqrt(np.mean(err**2))), mae=float(err.mean()), p95_abs=float(np.quantile(err,.95)))
    # Relative distance metrics are undefined at zero and not claimed for signed coordinates.
    valid = y > 1e-8
    result['relative_n'] = int(valid.sum()) if target_mode != 'signed' else 0
    result['rdb_percent'] = float(100*np.mean(err[valid]/y[valid])) if result['relative_n'] else None
    result['qr_percent'] = float(100*np.mean(err[valid]/y[valid] <= .05)) if result['relative_n'] else None
    return result


def split_and_scale(frame, cfg, seed):
    n = len(frame)
    rng = np.random.default_rng(seed)
    ids = np.arange(n)
    limit = cfg.get('max_rows')
    if limit and limit < n:
        if cfg['protocol'] != 'random' or cfg.get('window',1) > 1:
            raise ValueError('max_rows is supported only for snapshot random smoke tests.')
        ids = np.sort(rng.choice(ids, limit, replace=False))
    protocol = cfg['protocol']
    if protocol == 'group':
        col = cfg.get('group_column','well_id')
        if col not in frame or frame[col].isna().any():
            raise ValueError(f'Group protocol requires complete {col}.')
        groups = list(frame.iloc[ids][col].unique())
        if len(groups) < 3:
            raise ValueError('At least three independent groups are needed.')
        if 'test_group' in cfg:
            te, va = cfg['test_group'], cfg['val_group']
            if te == va or te not in groups or va not in groups:
                raise ValueError('Validation and test groups must exist and differ.')
        else:
            groups = rng.permutation(groups); te, va = groups[-1], groups[-2]
        masks = [~frame[col].isin([te,va]), frame[col]==va, frame[col]==te]
        parts = [ids[mask.to_numpy()[ids]] for mask in masks]
    elif protocol == 'chronological':
        if not {'well_id','time'}.issubset(frame.columns):
            raise ValueError('Chronological split needs well_id and numeric time.')
        parts = [[],[],[]]
        for _, group in frame.iloc[ids].groupby('well_id',sort=False):
            order = group.sort_values('time').index.to_numpy()
            cuts = [int(.6*len(order)), int(.8*len(order))]
            for dest, block in zip(parts,np.split(order,cuts)): dest.extend(block)
        parts = [np.array(x,dtype=int) for x in parts]
    elif protocol == 'random':
        if not cfg.get('allow_exploratory_random',False):
            raise ValueError('Random splits need explicit allow_exploratory_random=true.')
        ids = rng.permutation(ids); parts = np.split(ids,[int(.6*len(ids)),int(.8*len(ids))])
    else:
        raise ValueError('protocol must be random, group, or chronological.')
    if min(map(len,parts)) < 2: raise ValueError('Insufficient rows in a split.')
    # An augmentation source must never span partitions when provenance is supplied.
    if 'source_id' in frame:
        sources = [set(frame.iloc[x].source_id) for x in parts]
        if any(sources[i]&sources[j] for i in range(3) for j in range(i)):
            raise ValueError('Augmentation source_id crosses splits; use source-disjoint groups.')
    raw = frame[FEATURES].to_numpy(float); target = frame[TARGET].to_numpy(float)
    mean = raw[parts[0]].mean(0); scale = raw[parts[0]].std(0)
    scale = np.where(scale<1e-8,1.,scale)
    ym, ys = float(target[parts[0]].mean()), max(float(target[parts[0]].std()),1e-8)
    z = (raw-mean)/scale
    window = int(cfg.get('window',1))
    if window != 1 and (window < 2 or window % 2): raise ValueError('Use window=1 or an even window >=2.')
    packs, window_rows = [], []
    for part in parts:
        wi = build_windows(frame, part, window, cfg)
        last = wi[:,-1]
        packs.append(dict(x=z[wi].astype('float32'), inc=raw[last,9].astype('float32'),
                          y=((target[last]-ym)/ys).astype('float32'), rows=last))
        window_rows.append(wi.tolist())
    meta = dict(features=FEATURES,target=TARGET,target_mode=cfg['target_mode'],x_mean=mean.tolist(),x_scale=scale.tolist(),
                y_mean=ym,y_scale=ys,split_rows=dict(zip(['train','val','test'],[p.tolist() for p in parts])),
                window_rows=dict(zip(['train','val','test'],window_rows)),rows_total=n,rows_selected=len(ids),
                protocol=protocol,window=window,seed=seed)
    return packs,meta


def build_windows(frame, rows, window, cfg):
    if window == 1: return np.asarray(rows)[:,None]
    if cfg.get('protocol') == 'random' or not (cfg.get('verified_time',False) or cfg.get('sample_order_confirmed',False)):
        raise ValueError('Temporal branch requires verified_time or sample_order_confirmed and a nonrandom split.')
    if not {'well_id','time'}.issubset(frame.columns): raise ValueError('Missing well_id/time.')
    if frame[['well_id','time']].isna().any().any(): raise ValueError('Missing temporal metadata.')
    result=[]
    for _, group in frame.iloc[rows].groupby('well_id',sort=False):
        group=group.sort_values('time'); t=group.time.to_numpy(float)
        if not np.isfinite(t).all() or (np.diff(t)<=0).any(): raise ValueError('Time must increase strictly within a well.')
        if len(t)>2 and not np.allclose(np.diff(t),np.diff(t)[0],rtol=1e-4,atol=1e-8):
            raise ValueError('Uniform sampling required; resample explicitly before training.')
        order=group.index.to_numpy()
        result.extend(order[j-window+1:j+1] for j in range(window-1,len(order)))
    if not result: raise ValueError('No complete within-well, within-split windows.')
    return np.asarray(result)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
