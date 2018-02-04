import numpy as np
import pandas as pd
import time


def write_out_file(sessionid, outputs, states, out_file_output, out_file_state):
    for i in range(0, len(sessionid)):
        sid = sessionid[i]
        output = outputs[i]

        out_file_output.write(str(sid))
        for l in output:
            out_file_output.write("\t%f" % l)
        out_file_output.write("\n")
        for j in range(0, len(states)):
            state = states[j][i]
            out_file_state.write(str(sid))
            for k in state:
                out_file_state.write("\t%f" % k)
            out_file_state.write("\n")


def write_pred_state_file(change_idx, sessionid, states, preds, out_file_state, out_file_preds):
    for i in range(0, len(change_idx)):
        sid = str(sessionid[i])
        out_file_preds.write(sid)
        idx = change_idx[i]
        pred = preds[i]
        for l in pred:
            out_file_preds.write("\t%s" % l)
        out_file_preds.write("\n")

        for j in range(0, len(states)):
            state = states[j][idx]
            out_file_state.write(sid)
            for k in state:
                # print 'K:'
                # print k
                out_file_state.write("\t%f" % k)
            out_file_state.write("\n")

#
# def pred_sessions_batch(model, train_data, test_data, outfile, cut_off=5, batch_size=50, session_key='SessionId',
#                         item_key='ItemId', time_key='Time'):
#     out_file_output = open(outfile + ".out", "w")
#     out_file_state = open(outfile + ".state", "w")
#
#     model.predict = False
#     # Build itemidmap from train data.
#     itemids = train_data[item_key].unique()
#     itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
#
#     test_data.sort_values([session_key, time_key], inplace=True)
#     offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
#     offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
#     evalutation_point_count = 0
#     mrr, recall = 0.0, 0.0
#     if len(offset_sessions) - 1 < batch_size:
#         batch_size = len(offset_sessions) - 1
#     iters = np.arange(batch_size).astype(np.int32)
#     maxiter = iters.max()
#     start = offset_sessions[iters]
#     end = offset_sessions[iters + 1]
#     in_idx = np.zeros(batch_size, dtype=np.int32)
#     np.random.seed(42)
#     while True:
#         valid_mask = iters >= 0
#         if valid_mask.sum() == 0:
#             break
#         start_valid = start[valid_mask]
#         minlen = (end[valid_mask] - start_valid).min()
#         in_idx[valid_mask] = test_data[item_key].values[start_valid]
#
#         for i in range(minlen - 1):
#             out_idx = test_data[item_key].values[start_valid + i + 1]
#             ret_sessionid, ret_out, ret_state = model.output_batch(iters, in_idx, itemidmap, batch_size)
#             # print ret_sessionid
#
#             write_out_file(ret_sessionid, ret_out, ret_state, out_file_output, out_file_state)
#             in_idx[valid_mask] = out_idx
#
#         start = start + minlen - 1
#         mask = np.arange(len(iters))[(valid_mask) & (end - start <= 1)]
#         for idx in mask:
#             maxiter += 1
#             if maxiter >= len(offset_sessions) - 1:
#                 iters[idx] = -1
#             else:
#                 iters[idx] = maxiter
#                 start[idx] = offset_sessions[maxiter]
#                 end[idx] = offset_sessions[maxiter + 1]


def pred_sessions_batch(model, train_data, test_data, outfile, cut_off=5, batch_size=50, session_key='uid',
                        item_key='item_id', time_key='time_id'):
    out_file_state = open(outfile + ".state", "w")
    out_file_pred = open(outfile + ".pred", "w")

    model.predict = False
    # Build itemidmap from train data.
    itemids = train_data[item_key].unique()
    itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)

    test_data.sort_values([session_key, time_key], inplace=True)
    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    sessionidmap = test_data[session_key].unique()

    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1
    iters = np.arange(batch_size).astype(np.int32)
    maxiter = iters.max()
    start = offset_sessions[iters]
    end = offset_sessions[iters + 1]
    in_idx = np.zeros(batch_size, dtype=np.int32)
    np.random.seed(42)
    last_iters = []
    while True:
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        minlen = (end[valid_mask] - start_valid).min()
        in_idx[valid_mask] = test_data[item_key].values[start_valid]
        # print start_valid,end[valid_mask]
        # print 'minlen-------',minlen
        st = time.time()
        for i in range(minlen):
            # 跑了两遍初始的state零矩阵变为非零矩阵再计算了一次
            out_idx = test_data[item_key].values[start_valid + i]
            ret_pred, ret_state, ret_logit = model.pred_output_batch(iters, in_idx, itemidmap, batch_size)
            in_idx[valid_mask] = out_idx
            # print '--------in run ret_pred'
        last_iters = iters.copy()
        start = start + minlen
        ed = time.time()

        # print "step1 time :", ed-st


        mask = np.arange(len(iters))[(valid_mask) & (end - start <= 0)]
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions) - 1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter + 1]
        change_idx = np.arange(batch_size)[last_iters != iters]
        ed = time.time()

        # print "step2 time :", ed-st

        if len(change_idx) > 0:
            # print last_iters[change_idx],sessionidmap.shape,sessionidmap
            sessionid = sessionidmap[last_iters[change_idx]]
            # print 'change_idx----',change_idx,'sessionid-----',sessionid
            # print 'ret_pred',ret_pred.shape,ret_pred[change_idx]
            # print 'ret_state-----',ret_state[-1][change_idx]
            # print 'ret_logit=====',ret_logit[change_idx]

            index_sort_pred = np.argsort(-ret_pred[change_idx], axis=1)[:, 0:cut_off]
            # print '---index_sort_pred',index_sort_pred
            # print '---index_sort_logit',np.argsort(-ret_logit[change_idx],axis=1)[:,0:cut_off]
            ed = time.time()

            # print "step2.1 time :", ed-st
            pred_item = itemids[index_sort_pred]
            # print 'ret_logit=====',ret_pred[change_idx[0]][index_sort_pred[change_idx][0]]
            # print 'ret_logit=====',ret_logit[change_idx[0]][index_sort_pred[change_idx][0]]
            # print '----pred_item',pred_item
            write_pred_state_file(change_idx, sessionid, ret_state, pred_item, out_file_state, out_file_pred)
            ed = time.time()

            # print "step3 time :", ed-st
