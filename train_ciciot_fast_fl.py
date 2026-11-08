from __future__ import annotations
import argparse, gc, json, os, random, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'independent_fl_v1' / 'CICIoT2023'
MODEL_DIR = ROOT / 'models' / 'independent_fl_v1' / 'CICIoT2023'
RESULT_DIR = ROOT / 'results' / 'independent_fl_v1' / 'CICIoT2023'
CHUNK = 100_000
SEED = 42


def setup(seed=SEED):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass


def build_model(n_features, n_classes, lr):
    # Fast tabular classifier; no recurrent/attention layers.
    inp = tf.keras.Input(shape=(n_features,), name='iot_features')
    x = tf.keras.layers.Dense(256, activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inp, out, name='CICIoT_Fast_Federated_MLP')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def fit_scaler(files, features):
    count=0; mean=np.zeros(len(features), np.float64); m2=np.zeros(len(features), np.float64)
    for file in files:
        for df in pd.read_csv(file, usecols=features, chunksize=CHUNK, low_memory=False):
            a=df.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float64)
            n=len(a)
            if not n: continue
            bm=a.mean(0); bm2=((a-bm)**2).sum(0)
            if count==0: mean=bm; m2=bm2; count=n
            else:
                delta=bm-mean; total=count+n
                m2 += bm2 + delta*delta*count*n/total
                mean += delta*n/total; count=total
    std=np.sqrt(m2/max(count,1)); std[std<1e-12]=1.0
    return mean.astype('float32'), std.astype('float32')


def cache_client(csv_file, cache_folder, features, mean, std, n_classes):
    cache_folder.mkdir(parents=True, exist_ok=True)
    # Count rows first; these are the exact rows from the prepared train split.
    rows=sum(len(x) for x in pd.read_csv(csv_file, usecols=['label_id'], chunksize=CHUNK))
    X=np.memmap(cache_folder/'X.dat', mode='w+', dtype='float32', shape=(rows,len(features)))
    y=np.memmap(cache_folder/'y.dat', mode='w+', dtype='int32', shape=(rows,))
    pos=0
    for df in pd.read_csv(csv_file, usecols=features+['label_id'], chunksize=CHUNK, low_memory=False):
        xx=df[features].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy('float32')
        xx=(xx-mean)/std; np.clip(xx,-15,15,out=xx)
        yy=df['label_id'].to_numpy('int32')
        if np.any((yy<0)|(yy>=n_classes)): raise ValueError('Invalid label_id found in '+str(csv_file))
        end=pos+len(yy); X[pos:end]=xx; y[pos:end]=yy; pos=end
    X.flush(); y.flush()
    indices=[np.flatnonzero(np.asarray(y)==i) for i in range(n_classes)]
    counts=np.array([len(z) for z in indices], dtype=np.int64)
    # Classes absent from a particular client keep probability zero; the
    # global class list remains identical across all clients.
    # Moderate minority oversampling; not full uniform oversampling.
    probs=np.sqrt(counts.astype('float64')); probs/=probs.sum()
    return {'X':X,'y':y,'indices':indices,'probs':probs,'rows':rows,'counts':counts}


def batch_generator(cache, batch_size, seed):
    rng=np.random.default_rng(seed); nclasses=len(cache['indices'])
    while True:
        chosen=rng.choice(nclasses, size=batch_size, p=cache['probs'])
        ix=np.empty(batch_size, dtype=np.int64)
        for c in np.unique(chosen):
            where=np.flatnonzero(chosen==c)
            ix[where]=rng.choice(cache['indices'][int(c)], size=len(where), replace=True)
        rng.shuffle(ix)
        yield np.asarray(cache['X'][ix]), np.asarray(cache['y'][ix])


def fedavg(weight_sets, sizes):
    total=float(sum(sizes)); out=[]
    for k in range(len(weight_sets[0])):
        acc=np.zeros_like(weight_sets[0][k], dtype=np.float64)
        for w,s in zip(weight_sets,sizes): acc += w[k].astype(np.float64)*(s/total)
        out.append(acc.astype(weight_sets[0][k].dtype))
    return out


def evaluate(model, csv_file, features, mean, std, batch_size, n_classes):
    ys=[]; ps=[]; total_loss=0.; total=0
    for df in pd.read_csv(csv_file, usecols=features+['label_id'], chunksize=max(batch_size*100,10000), low_memory=False):
        xx=df[features].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy('float32')
        xx=(xx-mean)/std; np.clip(xx,-15,15,out=xx)
        yy=df['label_id'].to_numpy('int32')
        probs=model.predict(xx, batch_size=batch_size, verbose=0)
        pred=probs.argmax(1)
        losses=tf.keras.losses.sparse_categorical_crossentropy(yy,probs).numpy()
        total_loss += float(losses.sum()); total += len(yy)
        ys.append(yy); ps.append(pred.astype('int32'))
    y=np.concatenate(ys); p=np.concatenate(ps)
    cm=confusion_matrix(y,p,labels=np.arange(n_classes))
    return {'loss':total_loss/total,'accuracy':accuracy_score(y,p),
            'macro_f1':f1_score(y,p,average='macro',zero_division=0),
            'weighted_f1':f1_score(y,p,average='weighted',zero_division=0),
            'y':y,'pred':p,'cm':cm}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rounds',type=int,default=5)
    ap.add_argument('--local-steps',type=int,default=200, help='updates per client per round')
    ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--learning-rate',type=float,default=0.0005)
    ap.add_argument('--seed',type=int,default=SEED)
    args=ap.parse_args(); setup(args.seed)
    meta=json.loads((DATA_DIR/'metadata.json').read_text(encoding='utf-8'))
    features=meta['feature_columns']; classes=meta['class_names']; nc=len(classes)
    client_files=[DATA_DIR/f'train_client_{i:02d}.csv' for i in range(1,4)]
    for f in client_files+[DATA_DIR/'validation.csv', DATA_DIR/'test.csv']:
        if not f.exists(): raise FileNotFoundError(f)
    MODEL_DIR.mkdir(parents=True,exist_ok=True); RESULT_DIR.mkdir(parents=True,exist_ok=True)
    mean,std=fit_scaler(client_files,features)
    np.savez(DATA_DIR/'scaler_fast_mlp.npz',mean=mean,std=std,features=np.asarray(features,dtype=str))
    sizes=[]
    with tempfile.TemporaryDirectory(prefix='ciciot_fast_fl_') as tmp:
        caches=[]
        for i,f in enumerate(client_files,1):
            print(f'Preparing client {i}/3 cache...',flush=True)
            c=cache_client(f,Path(tmp)/f'client_{i}',features,mean,std,nc)
            caches.append(c); sizes.append(c['rows'])
            print(f'  rows={c["rows"]:,}; class counts={c["counts"].tolist()}',flush=True)
        model=build_model(len(features),nc,args.learning_rate)
        best=-1.; best_round=0; history=[]
        for rnd in range(1,args.rounds+1):
            print(f'\nFederated round {rnd}/{args.rounds}',flush=True)
            local_weights=[]
            for cid,c in enumerate(caches,1):
                local=build_model(len(features),nc,args.learning_rate)
                local.set_weights(model.get_weights())
                gen=batch_generator(c,args.batch_size,args.seed+rnd*100+cid)
                running=0.
                for step in range(args.local_steps):
                    xb,yb=next(gen)
                    result=local.train_on_batch(xb,yb,return_dict=True)
                    running += float(result['loss'])
                local_weights.append(local.get_weights())
                print(f'  Client {cid}: mean_loss={running/args.local_steps:.5f}; updates={args.local_steps}',flush=True)
                del local,gen; gc.collect()
            model.set_weights(fedavg(local_weights,sizes))
            val=evaluate(model,DATA_DIR/'validation.csv',features,mean,std,args.batch_size,nc)
            row={'round':rnd,'validation_loss':val['loss'],'validation_accuracy':val['accuracy'],
                 'validation_macro_f1':val['macro_f1'],'validation_weighted_f1':val['weighted_f1']}
            history.append(row); pd.DataFrame(history).to_csv(RESULT_DIR/'fast_mlp_round_history.csv',index=False)
            print(f"  Validation: loss={val['loss']:.5f}, accuracy={val['accuracy']:.5%}, macro_F1={val['macro_f1']:.5f}",flush=True)
            if val['macro_f1']>best:
                best=val['macro_f1']; best_round=rnd
                model.save(MODEL_DIR/'ciciot_fast_federated_mlp.keras')
                print('  Saved best validation macro-F1 checkpoint.',flush=True)
        for c in caches:
            c['X'].flush(); c['y'].flush(); del c['X']; del c['y']
        del caches; gc.collect()
    best_model=tf.keras.models.load_model(MODEL_DIR/'ciciot_fast_federated_mlp.keras',compile=False)
    test=evaluate(best_model,DATA_DIR/'test.csv',features,mean,std,args.batch_size,nc)
    metrics={k:v for k,v in test.items() if k not in ('y','pred','cm')}
    metrics.update({'dataset':'CICIoT2023','model':'Fast tabular MLP','federated_method':'sample-count-weighted FedAvg, three clients',
                    'best_round':best_round,'rounds':args.rounds,'local_steps_per_client_per_round':args.local_steps,
                    'batch_size':args.batch_size,'learning_rate':args.learning_rate,
                    'test_records':int(len(test['y'])),'number_of_classes':nc,
                    'scaler_fit':'training clients only'})
    (RESULT_DIR/'fast_mlp_test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    pd.DataFrame(classification_report(test['y'],test['pred'],labels=np.arange(nc),target_names=classes,output_dict=True,zero_division=0)).T.to_csv(RESULT_DIR/'fast_mlp_classification_report.csv')
    pd.DataFrame(test['cm'],index=classes,columns=classes).to_csv(RESULT_DIR/'fast_mlp_confusion_matrix.csv')
    print('\nFINAL UNTOUCHED TEST RESULTS'); print(json.dumps(metrics,indent=2))
    print('Model:',MODEL_DIR/'ciciot_fast_federated_mlp.keras')
    print('Metrics:',RESULT_DIR/'fast_mlp_test_metrics.json')
    tf.keras.backend.clear_session(); gc.collect()

if __name__=='__main__': main()
