from datetime import datetime
from re import findall
from ipaddress import ip_address
import pandas as pd

import pyximport
pyximport.install()
from pcap_constants import LIST_LENGTH, FLOW_TYPE_INSIDE_ONLY, \
                  FLOW_TYPE_OUTSIDE_ONLY, FLOW_TYPE_OUTGOING, \
                  FLOW_TYPE_INCOMING, FLOW_TYPE_UNIDENTIFIED

# formatters allow float because merge converts int to float to handle NaN
# epoch time displayed is UTC

def epoch_to_text(epoch):
    return datetime.fromtimestamp(epoch).strftime('%Y-%m-%d %H:%M:%S.%f')

def epoch_to_filename(epoch):
    return datetime.fromtimestamp(epoch).strftime('%Y-%m-%d_%H_%M_%S.%f')

# throws ValueError, use, e.g. "2021-10-26_15_26_58.478380"
def filename_to_epoch(filename):
    return datetime.strptime(filename, "%Y-%m-%d_%H_%M_%S.%f").timestamp()

# throws ValueError, use hh_mm_ss.decimal, e.g. "15_26_58.478380"
def period_to_epoch(filename):
    dt = datetime.strptime(filename, "%H_%M_%S.%f")
    return dt.hour * 60 * 60 + dt.minute * 60 + dt.second \
                              + dt.microsecond * 0.000001

def epoch_to_text_float(epoch):
    return "%.9f"%epoch

def ip_protocol_to_text(ip_protocol):
    if ip_protocol == 1:
        return "icmp"
    elif ip_protocol == 6:
        return "tcp"
    elif ip_protocol == 17:
        return "udp"
    elif ip_protocol == 50:
        return "esp"
    else:
        return "%d"%ip_protocol

# column formatter for printing compacted MAC
def mac_to_text(mac):
    return ":".join(findall("..", "%012x"%int(mac)))

# column formatter for printing IPs using df.to_string
def np_int_ip_to_text(np_int_ip):
    int_ip = int(np_int_ip.item())
    return ip_address(int_ip).__str__()

def flow_type_to_text(flow_type):
    if flow_type == FLOW_TYPE_INSIDE_ONLY:
        return "inside_only"
    if flow_type == FLOW_TYPE_OUTSIDE_ONLY:
        return "outside_only"
    if flow_type == FLOW_TYPE_OUTGOING:
        return "outgoing"
    if flow_type == FLOW_TYPE_INCOMING:
        return "incoming"
    if flow_type == FLOW_TYPE_UNIDENTIFIED:
        return "unidentified"
    raise RuntimeError("bad")

# usage: df.to_string(formatters = column_formatters))
column_formatters = {"timestamp":epoch_to_text,
                     "timestamp_float":epoch_to_text_float,
                     "mac":mac_to_text,
                     "mac_src":mac_to_text,
                     "mac_dst":mac_to_text,
                     "mac_src_inside":mac_to_text,
                     "mac_dst_inside":mac_to_text,
                     "mac_src_outside":mac_to_text,
                     "mac_dst_outside":mac_to_text,
                     "mac_src_primary":mac_to_text,
                     "mac_dst_primary":mac_to_text,
                     "mac_src_padded":mac_to_text,
                     "mac_dst_padded":mac_to_text,
                     "ip":np_int_ip_to_text,
                     "ip_src":np_int_ip_to_text,
                     "ip_dst":np_int_ip_to_text,
                     "ip_src_inside":np_int_ip_to_text,
                     "ip_dst_inside":np_int_ip_to_text,
                     "ip_src_outside":np_int_ip_to_text,
                     "ip_dst_outside":np_int_ip_to_text,
                     "ip_src_primary":np_int_ip_to_text,
                     "ip_dst_primary":np_int_ip_to_text,
                     "ip_src_padded":np_int_ip_to_text,
                     "ip_dst_padded":np_int_ip_to_text,
                     "inside_ip":np_int_ip_to_text,
                     "outside_ip":np_int_ip_to_text,
                     "ip_start":np_int_ip_to_text,
                     "ip_stop":np_int_ip_to_text,
                     "flow_type":flow_type_to_text,
                    }

def set_display_options(rows):
    pd.set_option("display.max_columns", None,
                  "display.max_rows", rows,"display.min_rows", None,
                  "display.max_colwidth", None, "display.width", None,
                  "display.precision", 18)
    pd.options.display.float_format = '{:.6f}'.format

