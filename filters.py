import os 
import h5py
import numpy as np
from scipy.sparse import csc_matrix

class ROISet:
    def __init__(self, cm_data, F_dff=None, snr=None, rval=None):
        # F_dff / snr / rval override the hdf5 values (OLL results.hdf5 can store
        # these as 'NoneType'; the reader passes in its computed/fallback arrays)
        #organizing pre-init
        data = cm_data['estimates']['A']['data']
        indices = cm_data['estimates']['A']['indices']
        indptr = cm_data['estimates']['A']['indptr']
        shape = cm_data['estimates']['A']['shape']
        self.y_dims = cm_data['estimates']['dims'][1]
        self.x_dims = cm_data['estimates']['dims'][0]
        self.F_dff = np.array(cm_data['estimates']['F_dff']) if F_dff is None else np.asarray(F_dff)
        self.snr = np.array(cm_data["estimates"]["SNR_comp"]) if snr is None else np.asarray(snr)
        self.rval = np.array(cm_data["estimates"]["r_values"]) if rval is None else np.asarray(rval)
        self.good_idxs =  np.array(cm_data['estimates']['idx_components'])
        
        sparse_mtx = csc_matrix((data, indices, indptr), shape=shape)
        rois = sparse_mtx.toarray()

        self.rois = np.array([i.reshape(self.y_dims, self.x_dims) for i in rois.T])  

    def prepass(self, rval_thresh=0.3):
        ogs = self.rois[self.good_idxs]
        ogrval = self.rval[self.good_idxs]
        mod_idx = np.array(list(range(self.good_idxs.shape[0])))
        mi_lookup = np.concatenate((mod_idx.reshape(-1,1), self.good_idxs.reshape(-1,1)), axis=1)
        passes = np.where(ogrval > rval_thresh)
        self.prepassidx = mi_lookup[passes] #FIRST FILTER: outputs col1: index of values ref to ogs, col2: ref to idx_components
        
        self.f1_rois = ogs[self.prepassidx[:,0]] #NOTE THIS PAIRING!!!
        
        
    def binarize(self):
        if self.f1_rois is None:
            use_rois = self.rois

        else:
            use_rois = self.f1_rois
    
        bft = np.array([np.where(i != 0, 1, 0) for i in use_rois], dtype='int8')
        nonzero = np.array([i.sum() for i in bft])
        self.bin_rois = bft
        self.areas = nonzero
        return self.bin_rois, self.areas
        
    def find_overlap(self):
##TO DO: UPDATE TO ALLOW VARIABLE INPUTS: DIFFERENT DIMENSIONS, ETC...
        if self.bin_rois is None:
            if self.f1_rois is None:
                use_rois = self.rois
            else:
                use_rois = self.f1_rois
        else:
            use_rois = self.bin_rois
            
##TODO: ADD VARIABLE USE CASE FOR AREAS--NEED TO GET AREA FROM GAUSSIAN, ETC.
        amtx = np.tile(self.areas, (self.areas.shape[0], 1))
        bmtx = np.tile(self.areas, (self.areas.shape[0], 1)).T
        plus_ab = amtx + bmtx

        # For binary masks (a_i + a_j - sum|p - q|) / 2 is the pixel intersection,
        # i.e. B @ B.T. Same result as the original N x N x H x W tiling, which
        # needs tens of GB for a few hundred ROIs on a 512 x 512 frame.
        n_pix = int(np.prod(use_rois.shape[1:]))
        flat = csc_matrix(use_rois.reshape(use_rois.shape[0], n_pix).astype(np.int32))
        invert_overlap = (flat @ flat.T).toarray()
        
        self.overlap_percents = np.round((invert_overlap / (plus_ab/2)), decimals=3)

    def filter_high_overlap(self, corr_thresh=0.2, overlap=0.4):
        if self.bin_rois is None:
            if self.f1_rois is None:
                use_rois = self.rois
            else:
                use_rois = self.f1_rois
        else:
            use_rois = self.bin_rois
     
        seek = np.where((self.overlap_percents>overlap) & (self.overlap_percents<1))

        pairwise_corr = np.corrcoef(self.F_dff[self.prepassidx[:,1],:])
        rvals = self.rval[self.prepassidx[:,1]] 
        snrs = self.snr[self.prepassidx[:,1]] 
        prefinal = []
        for i in np.unique(seek[0]):
            inseek = np.where(seek[0]==i)[0] #index of seek[1] where seek[0] is value i (row i)
            col = seek[1][inseek]
            ifallchecks = [i]
            for j in col:
                if pairwise_corr[i, j] > corr_thresh:
                    ifallchecks.append(j)

            bestpick = np.argmax(rvals[ifallchecks] * snrs[ifallchecks])
            worstpick = np.delete(ifallchecks, bestpick).tolist()

            for i in worstpick:
                prefinal.append(i)

        final = [i for i in range(len(self.prepassidx)) if i not in np.unique(prefinal)]
        fin_lookup = final
        self.best_rois = self.prepassidx[fin_lookup][:,1]
        return self.best_rois

















# RETIRED LOGIC:
# ##########################
# class ROISet:
#     def __init__(self, cm_data, rval_thresh=0.2):
#         #organizing pre-init
#         data = cm_data['estimates']['A']['data']
#         indices = cm_data['estimates']['A']['indices']
#         indptr = cm_data['estimates']['A']['indptr']
#         shape = cm_data['estimates']['A']['shape']
#         y_dims = cm_data['estimates']['dims'][1]
#         x_dims = cm_data['estimates']['dims'][0]
#         F_dff = np.array(cm_data['estimates']['F_dff'])
#         snr = np.array(cm_data["estimates"]["SNR_comp"])
#         rval = np.array(cm_data["estimates"]["r_values"])
#         to_use = list(cm_data["estimates"]["idx_components"])
#         interrval = rval[to_use]
#         to_use = [i for n, i in enumerate(to_use) if interrval[n] > rval_thresh]
#         sparse_mtx = csc_matrix((data, indices, indptr), shape=shape)
#         rois = sparse_mtx.toarray()
#         rois = rois[:, to_use]
        
#         #setting relevant vars
#         self.footprints = np.array([rois[:,i].reshape((x_dims, y_dims)) for i in range(rois.shape[1])])
#         self.snr = snr[to_use]
#         self.rval = rval[to_use]
#         self.F_dff = F_dff[to_use ,:]
#         self.to_use = to_use
        
#     def find_overlap(self, threshold=None):
#         footprints = self.footprints
#         bft = np.array([np.where(i != 0, 1, 0) for i in footprints], dtype='int8')
#         nonzero = [i.sum() for i in bft]
#         total = bft.shape[0]
#         overlap = np.zeros((total, total))
#         for n, i in enumerate(bft):
#             overlap[n, n:total] = [(np.multiply(i, bft[j]).sum())/((nonzero[n]+bft[j].sum())/2) for j in range(total) if j >= n] #SWAP OUT IF
#         if threshold is None:
#             self.overlap = overlap
#             return self.overlap
#         else:
#             overlap[overlap < float(threshold)] = 0
#             self.overlap = overlap
#             return self.overlap
            
#     def trace_correlate(self):
#         self.tracecorr = np.corrcoef(self.F_dff)
#         return self.tracecorr

#     def filter(self, threshold=None):
#         self.find_overlap(threshold=threshold)
#         self.trace_correlate()
#         self.cvo = np.multiply(self.tracecorr, self.overlap)

#         corr_over = np.where(self.cvo >= .9)#np.where(((self.cvo > 0.9) & (self.cvo <= 1)))
#         discounted_snr = np.multiply(self.snr, self.rval)

#         keep = []
#         for i in np.unique(corr_over[0]):
#             rois = [k[0] for k in corr_over[1][np.argwhere(corr_over[0]==i)].tolist()]
#             snrs = [discounted_snr[k] for k in rois]
#             keep.append(rois[snrs.index(max(snrs))])
            
#         sl = np.unique(keep).tolist()
#         self.best_rois = np.array(self.to_use)[sl]
#         return self.best_rois


























