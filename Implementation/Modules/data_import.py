# Imports packages
import openpyxl

# Reads All data
def read_data(data_file_path):
    
    #Opens the excel book
    book = openpyxl.load_workbook(data_file_path, data_only=True)


    # Generates Indices
    indices_sheet = book.get_sheet_by_name('Indices Data')

    # Generates T
    time_horizon = indices_sheet.cell(row=2, column=1).value
    t = [i+1 for i in range(time_horizon)]

    # Generates M
    tracked_wait = indices_sheet.cell(row=2, column=2).value
    m = [i for i in range(tracked_wait + 1)]

    # Generates P
    p = []
    for ppe_row in indices_sheet.iter_rows(min_row=2, min_col=3, max_col=3, values_only=True):
        if ppe_row[0] == None: break
        p.append(ppe_row[0])

    # Generates D
    d = []
    for comp_row in indices_sheet.iter_rows(min_row=2, min_col=4, max_col=4, values_only=True):
        if comp_row[0] == None: break
        d.append(comp_row[0])

    # Generates C
    c = []
    for cpu_row in indices_sheet.iter_rows(min_row=2, min_col=5, max_col=5, values_only=True):
        if cpu_row[0] == None: break
        c.append(cpu_row[0])


    # PPE Data
    ppe_data_sheet = book.get_sheet_by_name('PPE Data')

    ppe_data = {}
    for ppe_row in ppe_data_sheet.iter_rows(min_row=2, min_col=1, max_col=3, values_only=True):
        if ppe_row[0] == None: break

        ppe_data[ppe_row[0]] = {
            'expected units': ppe_row[1],
            'deviation': [int(i) for i in ppe_row[2].split(',')]
        }


    # PPE Usage
    usage_sheet = book.get_sheet_by_name('PPE Usage')

    usage = {}
    for usage_row in usage_sheet.iter_rows(min_row=2, min_col=1, max_col=4, values_only=True):
        if usage_row[0] == None: break

        usage[
            (usage_row[2], usage_row[1], usage_row[0])
        ] = usage_row[3]


    # Arrival Rates
    arrival_sheet = book.get_sheet_by_name('Patient Arrival')
    arrival = {}
    for arr_row in arrival_sheet.iter_rows(min_row=2, min_col=1, max_col=3, values_only=True):
        if arr_row[0] == None: break

        arrival[
            (arr_row[1], arr_row[0])
        ] = arr_row[2]


    # Patient Transitions
    transition_sheet = book.get_sheet_by_name('Patient Transitions')
    transition = {}
    for row in transition_sheet.iter_rows(min_row=2, min_col=1, max_col=4, values_only=True):
        if row[0] == None: break

        transition[
            (row[2], row[1], row[0])
        ] = row[3]


    # Model Parameters
    model_param_sheet = book.get_sheet_by_name('Model Parameters')
    cost_wait_val = model_param_sheet.cell(row=2, column=1).value
    cost_cancel = model_param_sheet.cell(row=2, column=2).value


    return (
        {'t': t, 'm': m, 'p': p, 'd': d, 'c': c },
        ppe_data,
        usage,
        arrival,
        transition,
        {'cw': cost_wait_val, 'cc': cost_cancel}
    )