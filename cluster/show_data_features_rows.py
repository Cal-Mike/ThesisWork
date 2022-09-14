#!/usr/bin/env python3
from argparse import ArgumentParser
from os.path import expanduser
import os
import numpy as np
import pandas as pd
from pandas_column_formatters import epoch_to_text, epoch_to_text_float, \
                             ip_protocol_to_text, \
                             np_int_ip_to_text, flow_type_to_text
from data_features_type import DATA_FEATURES_DTYPE

def _args():
    parser = ArgumentParser(
                     description="Show data features table rows")
    parser.add_argument("npy_file", type=str,
                     help="A .npy table file")
    args = parser.parse_args()
    return args

def _show_key(row):
    print("key:", row["key"])

def _show_flow_identifier(row):
    print("flow_identifier:",
          ip_protocol_to_text(row["ip_protocol"]),
          np_int_ip_to_text(row["ip_nps"]),
          np_int_ip_to_text(row["ip_other"]),
          row["port_nps"],
          row["port_other"],
         )

def _show_integer_totals(row):
    print("integer cumulative totals:",
          row["int_in_packets"], 
          row["int_in_bytes"], 
          row["int_out_packets"], 
          row["int_out_bytes"], 
         )

def _show_timestamps(row):
    print("timestamps: from %s to %s"%(
                            epoch_to_text(row["begin_timestamp"]),
                            epoch_to_text(row["end_timestamp"])))

#def _show_timestamps(row):
#    print("timestamps:",
#          epoch_to_text(row["begin_timestamp"]),
#          epoch_to_text(row["end_timestamp"]),
#          epoch_to_text_float(row["begin_timestamp"]),
#          epoch_to_text_float(row["end_timestamp"]),
#         )

def _show_packet_totals_in(row):
    print("packet totals in:",
          row["in_packets"],
          row["in_bytes"],
          row["in_mean_packet_length"],
          row["in_packets_per_second"],
          row["in_bytes_per_second"],
         )

def _show_packet_totals_out(row):
    print("packet totals out:",
          row["out_packets"],
          row["out_bytes"],
          row["out_mean_packet_length"],
          row["out_packets_per_second"],
          row["out_bytes_per_second"],
         )

def _show_list_totals_in(row):
    print("list totals in:",
          row["list_in_packets"],
          row["list_in_bytes"],
          row["list_in_mean_packet_length"],
          row["list_in_packets_per_second"],
          row["list_in_bytes_per_second"],
         )

def _show_list_totals_out(row):
    print("list totals out:",
          row["list_out_packets"],
          row["list_out_bytes"],
          row["list_out_mean_packet_length"],
          row["list_out_packets_per_second"],
          row["list_out_bytes_per_second"],
         )

def _show_lists(row):
    print("flow_types", row["flow_types"])
    print("delta_timestamps", row["delta_timestamps"])
    print("packet_lengths", row["packet_lengths"])

def _show_rows(np_table):
    for i in range(10):
        print("\nrow %d: "%i)
        _show_key(np_table[i])
        _show_flow_identifier(np_table[i])
        _show_integer_totals(np_table[i])
        _show_timestamps(np_table[i])
        _show_packet_totals_in(np_table[i])
        _show_packet_totals_out(np_table[i])
        _show_list_totals_in(np_table[i])
        _show_list_totals_out(np_table[i])
        _show_lists(np_table[i])

def _show_min_stats(np_table):
    for i in range(np_table.shape[0]):
        row = np_table[i]
        print("%d %s %s %s %s %s  %d in (%d), %d out (%d)"
              %(i,
                ip_protocol_to_text(row["ip_protocol"]),
                np_int_ip_to_text(row["ip_nps"]),
                np_int_ip_to_text(row["ip_other"]),
                row["port_nps"],
                row["port_other"],
                row["int_in_packets"], 
                row["int_in_bytes"], 
                row["int_out_packets"], 
                row["int_out_bytes"], 
             ))

if __name__ == "__main__":
    #args = _args()
    #npy_file = expanduser(args.npy_file)
    CLUST_PATH_MAY = '/home/bdallen/work_new/pcap_data_features_table_may_6-7/' #PATH to clustering dataset directory
    dir_list = os.listdir(CLUST_PATH_MAY)
    npy_file = CLUST_PATH_MAY+dir_list[0]
    #npy_table = np.load(CLUST_PATH_MAY+dir_list[0])

    npy_table = np.load(npy_file)
    memoryview_table = memoryview(npy_table)
    np_table = np.frombuffer(memoryview_table, dtype=DATA_FEATURES_DTYPE)
    print("file %s shape"%npy_file, np_table.shape)
    #_show_min_stats(np_table)
    #_show_rows(np_table)

    import socket, struct

    data_features_list = [item[0] for item in DATA_FEATURES_DTYPE]

    df = pd.DataFrame(columns=data_features_list)
    for i in range(10):
        row = np_table[i]
        df.loc[i] = [row[item] for item in data_features_list]
    print(df)

