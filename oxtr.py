'''
class OXTR

created: July 2022
author: Jay Pi, 2020jaypi@gmail.com
'''

import os, ast, openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
import seaborn as sns
import matplotlib.cm as cm
from scipy.stats import mannwhitneyu
from datetime import datetime as dt


class oxtr:
    def __init__(self, is_multi, path_data):
        self.is_multi = is_multi
        if self.is_multi:
            self.path_data = path_data + 'multi_intro/'
            self.mode_smoothing = 'nearest'
        else:
            self.path_data = path_data + 'single_intro/'
            self.mode_smoothing = 'interp'
        self.path_cb = self.path_data + 'cellbase/'
        self.path_analysis = self.path_data + 'analysis/'
        self.path_fig = self.path_analysis + 'figs/'     
        Path(self.path_data).mkdir(parents=True, exist_ok=True)    
        Path(self.path_cb).mkdir(parents=True, exist_ok=True)
        Path(self.path_analysis).mkdir(parents=True, exist_ok=True)
        Path(self.path_fig).mkdir(parents=True, exist_ok=True)

### ----- Methods ----- ###
    def _1_preprocess(self):
        '''preprocess'''
        print(f'preprocess -- {dt.now()}')
        fname_ev_ts = 'multi_introductions.csv' if self.is_multi else 'TimeOfIntroduction_allexpts.csv'
        # 1) save a cleaned event file & cleaned data
        df_ev = pd.read_csv(self.path_data+fname_ev_ts)
        ev = self.preprocess_ev_ts(df_ev)
        self.save_cellid_cleaned_data()
        # 2) save event-aligned Ca2+ trace data for each cell
        df_id = pd.read_csv(self.path_cb+'list_cellid.csv')
        list_ids = df_id.id.tolist()
        list_mouse_sess = df_id.mouse_sess.unique()
        for ms in list_mouse_sess:
            ms = ms.split('_')
            m, s = ms[0], ms[1]
            df = pd.read_csv(self.path_cb+f'{m}/{s}/{m}_{s}.csv')
            time_ = df.time_                   
            for c in df.columns[1:]: # 0 = time_
                dFF = df.loc[:, c]
                cellid = f'{m}_{s}_{c}'
                self.save_cell_trace(cellid, time_, dFF)

    def _2_plot_traces(self):
        ''' plot event-aligned Ca2+ traces; both overlaid & individual'''
        print(f'traces -- {dt.now()}')
        # 1) preprocess
        df_ev = pd.read_csv(self.path_cb+'event_ts.csv')
        list_ev = df_ev.columns[2:].tolist()
        list_ev = [e for e in list_ev if 'start_' not in e] # remove 'start_'
        df_id = pd.read_csv(self.path_cb+'list_cellid.csv')
        list_ids = df_id.id.tolist()
        list_mouse_sess = df_id.mouse_sess.unique()
        # 2) overlaid
        for ev_nm in list_ev:
            for ms in list_mouse_sess:
                str_ = ms.split('_')
                mouse, sess = str_[0], str_[1]
                self.plot_overlaid_ev(mouse, sess, ev_nm)
        # 3) single cell
        for ev_nm in list_ev:
            for ms in list_mouse_sess:
                ms = ms.split('_')
                m, s = ms[0], ms[1]
                df = pd.read_csv(self.path_cb+f'{m}/{s}/{m}_{s}.csv')
                self.save_each_cell_ev_trace(m, s, ev_nm)

    def _3_detect_peaks(self, tp=40, flg_plot=False):
        '''detect peaks'''
        print(f'peak detection -- {dt.now()}')
        # 1) list_cellid & event_ts
        df_ev = pd.read_csv(self.path_cb+'event_ts.csv')
        list_ev = df_ev.columns[2:].tolist()
        list_ev = [e for e in list_ev if 'start_' not in e] # remove 'start_'
        df_id = pd.read_csv(self.path_cb+'list_cellid.csv')
        list_ids = df_id['id'].tolist()
        list_mouse_sess = df_id.mouse_sess.unique()
        df_id.set_index('id', inplace=True)
        # 2) peak & trough detection 
        for ev in list_ev:
            # 1) init -- dtype should be object to store list in a cell
            for x in ['pk', 'tr']:
                df_id[f'{x}_ts_neg40_{ev}'], df_id[f'{x}_ts_pos40_{ev}'], df_id[f'{x}_amp_neg40_{ev}'], \
                df_id[f'{x}_amp_pos40_{ev}'], df_id[f'{x}_cnt_neg40_{ev}'], df_id[f'{x}_cnt_pos40_{ev}'] \
                = None, None, None, None, None, None
                df_id[f'{x}_ts_neg40_{ev}'] = df_id[f'{x}_ts_neg40_{ev}'].astype('object')
                df_id[f'{x}_ts_pos40_{ev}'] = df_id[f'{x}_ts_pos40_{ev}'].astype('object')
                df_id[f'{x}_amp_neg40_{ev}'] = df_id[f'{x}_amp_neg40_{ev}'].astype('object')
                df_id[f'{x}_amp_pos40_{ev}'] = df_id[f'{x}_amp_pos40_{ev}'].astype('object')
                df_id[f'{x}_cnt_neg40_{ev}'] = df_id[f'{x}_cnt_neg40_{ev}'].astype('object')
                df_id[f'{x}_cnt_pos40_{ev}'] = df_id[f'{x}_cnt_pos40_{ev}'].astype('object')
            # 2) calculation for each cell 
            for id_ in df_id.index:
                m, s, c = self.parse_cellid(id_)
                df = pd.read_csv(self.path_cb+f'{m}/{s}/{ev}/{m}_{s}_{c}_{ev}.csv')
                time_, dFF = df.time_ ,df.dFF
                bl_pk2pk = self.get_bl_pk2pk(dFF)
                ind_pk, ind_tr, z_dFF = self.detect_peak_trough(dFF, bl_pk2pk)
                list_pkts = time_[ind_pk].values
                list_trts = time_[ind_tr].values
                # peaks & troughs
                for x in ['pk', 'tr']:
                    list_ = list_pkts if x=='pk' else list_trts
                    ind_neg40 = np.where((list_>=-tp)&(list_<0))[0]  # ############### I changed here; 40 to tp
                    ind_pos40 = np.where((list_<tp)&(list_>=0))[0] # ############### I changed here 40; to tp
                    df_id.at[id_, f'{x}_ts_neg40_{ev}'] = list(time_[ind_neg40])
                    df_id.at[id_, f'{x}_ts_pos40_{ev}'] = list(time_[ind_pos40])
                    df_id.at[id_, f'{x}_amp_neg40_{ev}'] = list(dFF[ind_neg40])
                    df_id.at[id_, f'{x}_amp_pos40_{ev}'] = list(dFF[ind_pos40])
                    df_id.at[id_, f'{x}_cnt_neg40_{ev}'] = len(ind_neg40)
                    df_id.at[id_, f'{x}_cnt_pos40_{ev}'] = len(ind_pos40)
                # plot
                if flg_plot:
                    self.plot_detected_peaks(time_, dFF, z_dFF, ind_pk, f'{m}_{s}_{c}_{len(ind_pk)}pks')
            # save
            df_id.to_excel(self.path_analysis+'data_pk_tr_info.xlsx')

    
    
    

    


### --------------------- ###
    def set_dt(self, s):
        self.dt = np.diff(s)[1]

    def preprocess_cleaning(self, df_):
        ''' cleaning data '''
        df = df_.copy()
        df.columns = df.columns.str.replace(' ','')
        df.iloc[0, :] = df.iloc[0, :].str.strip() # ' accepted' -> 'accepted'
        df.rename(columns={'':'time_'}, inplace=True)
        # remove 'rejected' or 'not determined' cells
        for c in df.columns[1:]: # keep 'time_'
            if (df.loc[0, c] != 'accepted'):
                df.drop(columns=[c], inplace=True) 
        # remove the 1st row w/ the values 'accepted' & 'rejected'            
        df.drop(0, inplace=True) 
        # change dtype from str to float
        for c in df.columns:
            df[c] = df[c].astype(np.float)
        return df

    def preprocess_ev_ts(self, df_):
        # clean up and save events & timestamps
        df = df_.copy()
        df.dropna(subset='Data', inplace=True)
        df.columns = df.columns.str.replace(' ','')
        fname = df['Data'].str.split('_')
        for i in range(df.shape[0]):
            df.loc[i, 'mouse'] = fname[i][1].lower()
            df.loc[i, 'session'] = fname[i][0][2:]
        df['start_sess'] = 0
        if self.is_multi:
            cols_ = ['mouse', 'session', 'start_sess', 'strngr_in_1', 'strngr_out_1', 'strngr_in_2', 'strngr_out_2', 'strngr_in_3', 'strngr_out_3']
        else:
            cols_ = ['mouse', 'session', 'hab1', 'start_strngr_sess', 'intro_strngr', 'start_hab2', 'hab2', 'start_obj_sess', 'intro_obj', 'start_hab3', 'hab3', 'handwave']
        df[cols_].to_csv(self.path_cb+'event_ts.csv', index=False)
        return df[cols_]


    def load_ev_ts(self):
        ev = pd.read_csv(self.path_cb+'event_ts.csv') 
        ev.session = ev.session.astype(str) # just for session; str dtype is easier to handle
        return ev


    def save_cellid_cleaned_data(self):

        df_id = pd.DataFrame()
        list_animals = [m for m in os.listdir(self.path_cb) if os.path.isdir(self.path_cb+f'{m}')]
        list_animals.sort()

        cell_num = 0
        for m in list_animals:
            list_sess = []
            list_sess = [f for f in os.listdir(self.path_cb+f'{m}/') if os.path.isdir(self.path_cb+f'{m}/{f}')]
            list_sess.sort()
            
            for s in list_sess:
                path_sess = self.path_cb+f'{m}/{s}/'
                fname =  glob(path_sess + '*Strangers.csv') if self.is_multi else glob(path_sess + '*aftercontour.csv')
                df = pd.read_csv(fname[0])
                df = self.preprocess_cleaning(df)

                # save
                df.to_csv(path_sess+f'{m}_{s}.csv', index=False)

                time_ = df['time_']
                list_cells = df.columns.tolist()
                list_cells.remove('time_')
                for c in list_cells: # exclude 'time'
                    df_id.loc[cell_num, 'id'] = f'{m}_{s}_{c}'
                    df_id.loc[cell_num, 'mouse_sess'] = f'{m}_{s}'
                    df_id.loc[cell_num, 'mouse']  = f'{m}'
                    df_id.loc[cell_num, 'session']  = f'{s}'
                    df_id.loc[cell_num, 'cell'] = f'{c}'
                    cell_num += 1

                    # # save raw Ca2+ traces (time_ vs. dFF) for each cell in the session folder
                    # self.save_time_dFF_each_cell(path_sess, cellid, time_, df[c])
                    # 
                    
        # save list_cellid
        df_id.to_csv(self.path_cb+'list_cellid.csv', index=False)

    def load_sess_data(self, mouse, sess):
        fname = self.path_cb+f'{mouse}/{sess}/{mouse}_{sess}.csv'
        return pd.read_csv(fname)

    def plot_overlaid_ev(self, mouse, sess, ev_nm):
        #  load a session data & ev_ts
        df = self.load_sess_data(mouse, sess)
        df_ev = self.load_ev_ts()
        ev_ts = df_ev.loc[(df_ev.mouse==mouse)&(df_ev.session==sess), ev_nm].values[0] # ev_ts
        
        # set time windows & data
        t_offset = 60 #if self.is_multi else 60
        t_win = [ev_ts-t_offset, ev_ts, ev_ts+t_offset] # 2*
        t_win_zeroed = [ev_ts-t_offset, ev_ts, ev_ts+t_offset] - ev_ts #2*
        ind_win = [(df.time_>=t_win[0]).argmax(), (df.time_>=t_win[1]).argmax(), (df.time_>=t_win[2]).argmax()] 
        time_zeroed = df.loc[ind_win[0]:ind_win[2], 'time_'] - ev_ts
        time2plot = time_zeroed# [ind_win[0]:ind_win[2]+1]
        data = df.iloc[:, 1:]

        # plot
        fig, ax = plt.subplots(figsize=(12,4))
        for c in data.columns:
            dFF = data.loc[ind_win[0]:ind_win[2], c]
            ax.plot(time2plot, dFF)

        # cosmetic
        xlim, ylim = [t_win_zeroed[0], t_win_zeroed[2]], [-.1, .3]
        ax.plot([0, 0], [ylim[0], ylim[1]], '--', color='gray', linewidth=0.3)
        ax.plot([xlim[0], xlim[1]], [0, 0], color='gray', linewidth=1)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_yticks(np.arange(ylim[0], ylim[1], 0.1))
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('dF/F')
        ax.set_title(f'{ev_nm}_{mouse}_{sess}')
        path_save = self.path_data+f'analysis/overlaid/{mouse}_{sess}/'
        Path(path_save).mkdir(parents=True, exist_ok=True)
        fig.savefig(path_save+f'{ev_nm}_{mouse}_{sess}.eps', format='eps')
    
        

    def parse_cellid(self, cell_id):
        str_ = cell_id.split('_')
        return str_[0], f'{str_[1]}', str_[2] # mouse, sess, cell


    def save_cell_trace(self, cell_id, time_, dFF):
        # save Ca2+ traces (time_ vs. dFF) for each cell
        df_cell = pd.DataFrame()
        df_cell['time_'], df_cell['dFF']  = time_, dFF
        mouse, sess, cell = self.parse_cellid(cell_id)
        path_save = self.path_cb+f'{mouse}/{sess}/cells/'
        Path(path_save).mkdir(parents=True, exist_ok=True)
        df_cell.to_csv(path_save+f'{cell_id}.csv', index=False)


    def save_each_cell_ev_trace(self, mouse, sess, ev_nm):
        #  load a session data & ev_ts
        df = self.load_sess_data(mouse, sess)
        df_ev = self.load_ev_ts()
        ev_ts = df_ev.loc[(df_ev.mouse==mouse)&(df_ev.session==sess), ev_nm].values[0] # ev_ts

        # set time windows & data
        t_offset = 60
        t_win = [ev_ts-t_offset, ev_ts, ev_ts+2*t_offset]
        t_win_zeroed = [ev_ts-t_offset, ev_ts, ev_ts+2*t_offset] - ev_ts
        ind_win = [(df.time_>=t_win[0]).argmax(), (df.time_>=t_win[1]).argmax(), (df.time_>=t_win[2]).argmax()] 
        time_zeroed = df.loc[ind_win[0]:ind_win[2], 'time_'] - ev_ts
        time2plot = time_zeroed# [ind_win[0]:ind_win[2]+1]
        data = df.iloc[:, 1:]

        # plot
        for c in data.columns:
            dFF = data.loc[ind_win[0]:ind_win[2], c]

            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(time2plot, dFF)
            # cosmetic
            xlim, ylim = [t_win_zeroed[0], t_win_zeroed[2]], [-.1, .25]
            ax.plot([0, 0], [ylim[0], ylim[1]], '--', color='gray', linewidth=0.3)
            ax.plot([xlim[0], xlim[1]], [0, 0], color='gray', linewidth=1)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_yticks(np.arange(ylim[0], ylim[1], 0.1))
            ax.set_xlabel('time (sec)')
            ax.set_ylabel('dF/F')
            ax.set_title(f'{ev_nm}_{mouse}_{sess}')

            # save fig
            path_fig = self.path_analysis+f'individual/{mouse}_{sess}/'
            Path(path_fig).mkdir(parents=True, exist_ok=True)
            fig.savefig(path_fig+f'{ev_nm}_{mouse}_{sess}_{c}.eps', format='eps')

            # save each trace
            df_trace = pd.DataFrame()
            df_trace['time_'], df_trace['dFF'] = time2plot, dFF
            path_trace = self.path_cb+f'{mouse}/{sess}/{ev_nm}/'
            Path(path_trace).mkdir(parents=True, exist_ok=True)
            df_trace.to_csv(path_trace+f'{mouse}_{sess}_{c}_{ev_nm}.csv', index=False)
    

    def get_bl_pk2pk(self, dFF):
        # calculate a peak to peak value of baseline; a proxy of a noise level
        time_dur, n_pt10s = 18, 150 #10 sec=150 points when dt=0.06662899999999995
        list_pk2pk = []
        for n, tw in enumerate(np.arange(0, time_dur*n_pt10s, n_pt10s)): # 0 ~ 180sec
            trace_ = dFF[n*(n_pt10s):(n+1)*n_pt10s]
            pk2pk_ = np.max(trace_)-np.min(trace_)
            list_pk2pk.append(pk2pk_)
        list_pk2pk.sort()
        return np.mean(list_pk2pk[:5]) # mean value of the first lowest 5 values


    # ----- peak/trough detection-related --------
    def remove_redundant_peaks(self, ind_pk, dist_):
        ind2rm = np.where(np.diff(ind_pk)<dist_)[0]+1 # add 1 to deal with diff
        return np.delete(ind_pk, ind2rm)


    def detect_peak_trough(self, dFF, bl_pk2pk, height=3, min_pk_dist=10, n_pts=101, n_poly=3, smoothing=True):
        '''
        detect peaks & troughs and return indice & smoothed & z-scored dFF
        TODO: correct peak/trough indice (minor difference)
        '''
        # z-score (standardization); min_pk_dist 45 = 3sec
        mean_, std_ = dFF.mean(), dFF.std()
        z_dFF = (dFF-mean_)/bl_pk2pk 
        # smoothing
        if smoothing: z_dFF = savgol_filter(z_dFF, n_pts, n_poly, mode=self.mode_smoothing) 
        p_ind, _ = find_peaks(z_dFF, height) # peak, height*std 
        t_ind, _ = find_peaks(-z_dFF, height) # trough, height*std 
        p_ind = self.remove_redundant_peaks(p_ind, min_pk_dist)
        t_ind = self.remove_redundant_peaks(t_ind, min_pk_dist)
        return list(p_ind), list(t_ind), z_dFF

    def plot_detected_peaks(self, time_, dFF, dff, ind_pk, cellid):
        fig, ax = plt.subplots(2, 1, figsize=(12, 4))
        ax[0].title.set_text(cellid)
        ax[0].plot(time_, dFF)
        ax[0].plot(time_[ind_pk], dFF[ind_pk], 'x')
        ax[0].set_xlim(-60, 120)
        ax[0].set_ylim(-0.1, 0.25)
        ax[1].plot(time_, dff)
        ax[1].plot(time_[ind_pk], dff[ind_pk], 'x')
        ax[1].set_xlim(-60, 120)
        ax[1].set_xlabel('Time (s)')
    
    def plot_trace(self, time_, dFF):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_, dFF)
        ax.set_xlim(-60, 120)
        ax.set_xlabel('Time (s)')

        
