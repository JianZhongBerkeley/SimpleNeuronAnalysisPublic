import numpy as np

def get_clean_trace_dFF(src_template, src_spike_idxs, trace_len):
    template_len = len(src_template)
    template_spike_offset = np.argmax(src_template)
    
    src_spike_idxs[src_spike_idxs < template_spike_offset] = template_spike_offset
    src_spike_idxs[src_spike_idxs >= trace_len - template_spike_offset] = trace_len - template_spike_offset - 1
    
    src_spike_idxs = np.unique(src_spike_idxs)
    
    src_dFF_trace = np.zeros((trace_len,))
    
    for cur_spike_idx in src_spike_idxs:
        cur_spike_start_idx = cur_spike_idx - template_spike_offset
        src_dFF_trace[cur_spike_start_idx:cur_spike_start_idx+template_len] += src_template
    
    return src_dFF_trace, src_spike_idxs


def get_noise_trace_dFF(src_dFF_trace, nof_photons):

    src_F0 = 1
    src_F = src_F0 - src_dFF_trace
    
    dst_F = src_F * (nof_photons/np.mean(src_F))
    dst_F = np.random.poisson(lam = dst_F)
    
    dst_F0 = nof_photons
    dst_dF = dst_F - dst_F0 
    dst_dFF = dst_dF/dst_F0
    dst_dFF = -dst_dFF
    
    return dst_dFF