import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _normalize_data_table(data):
    mins = []
    for i in range(data.shape[1]):
        m = min(data[:, i])
        if m < 0:
            m *= -1.
        mins.append(m)

    mins = np.array(mins)

    data += np.tile(mins, (data.shape[0], 1,))
    data /= np.tile(data.max(axis=0), (data.shape[0], 1,))

    return data


# TODO: Add column and row labels
# TODO: Add highlight rows and columns
# TODO: normalize data columns and mark missing data with red
def plot_cc_model(data, model, model_logps, ax=None):
    n_cols = len(model['col_assignment'])
    n_rows = len(model['row_assignments'][0])

    data = _normalize_data_table(np.copy(data))

    col_asgn = model['col_assignment']
    c_start = 0

    data_mats = []

    gs = gridspec.GridSpec(n_rows, n_cols)

    view_order = np.argsort(model['view_counts'])[::-1]
    for vidx in view_order:
        cols_in_view = [i for i in range(n_cols) if col_asgn[i] == vidx]
        col_logps = [model_logps['column_logps'][i] for i in cols_in_view]
        cols = [cols_in_view[i] for i in np.argsort(col_logps)[::-1]]

        c_end = c_start + len(cols_in_view) - 1

        data_mats_view = []

        r_start = 0
        clstr_cts = np.bincount(model['row_assignments'][vidx])
        clstr_order = np.argsort(clstr_cts)[::-1]

        row_asgn = model['row_assignments'][vidx]
        for cidx in clstr_order:
            rows_in_cat = [j for j in range(n_rows) if row_asgn[j] == cidx]
            r_end = r_start + len(rows_in_cat) - 1

            row_logps = [model_logps['row_logps'][vidx][r] for r in rows_in_cat]
            sorted_row_idxs = np.argsort(row_logps)[::-1]
            rows = [rows_in_cat[i] for i in sorted_row_idxs]
            mat = np.zeros((len(rows), len(cols_in_view),))

            for i, row in enumerate(rows):
                for j, col in enumerate(cols):
                    mat[i, j] = data[row, col]

            ax = plt.subplot(gs[r_start:r_end+1, c_start:c_end+1])
            ax.matshow(mat, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            # data_mats_view.append(mat)

            r_start = r_end + 1

        data_mats.append(data_mats_view)

        c_start = c_end + 1
