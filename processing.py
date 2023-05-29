from time import gmtime, strftime
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pandas


# Custom Libraries
def create_events_dict(xml_text):
    """
    This function takes raw xml data from the exported_events.xml file
    and creates a dictionary of the data, to facilitate creation of
    either a Pandas DataFrame

    :param xml_text: raw xml data
    :return:
    """
    from lxml import etree

    data = []
    root = etree.fromstringlist(xml_text).getchildren()
    for field in root:
        row = {}
        dtime = long(field[0].text)
        row['datetime'] = strftime("%b %d, %Y %H:%M:%S", gmtime(dtime))
        row['host'] = field[1].text
        row['description'] = field[2].text
        row['lastvalue'] = field[3].text
        row['value'] = field[4].text
        data.append(row)

    return pandas.DataFrame(data, columns=["datetime", "host", "description", "lastvalue", "value"])


def ip2int(column):
    """
    Test
    :param column: Column from DataFrame
    :return:
    """
    addresses = []
    for i in range(len(column.values)):
        addr = str(column.values.item(i))
        if addr == 'nan':
            addresses.append(0)
        else:
            octals = addr.split('.')
            if octals.__len__() < 4:
                addresses.append(0)
            else:
                addresses.append((int(octals[0]) << 24) + (int(octals[1]) << 16) + (int(octals[2]) << 8) + int(octals[3]))

    return pandas.Series(addresses)


def macaddr2int(column):
    """
    Test
    :param column: Column from DataFrame
    :return:
    """
    addresses = []
    for i in range(len(column.values)):
        addr = str(column.values.item(i))
        if addr == 'nan':
            addresses.append(0)
        elif addr.__contains__("."):
            octals = addr.split('.')
            addresses.append((int(octals[0]) << 24) + (int(octals[1]) << 16) + (int(octals[2]) << 8) + int(octals[3]))
        else:
            segs = addr.split(':')
            addresses.append(int(segs[0], 16) + int(segs[1], 16) + int(segs[2], 16) \
                                 + int(segs[3], 16) + int(segs[4], 16) + int(segs[5], 16))
    return pandas.Series(addresses)


def format_datetime(column):
    """
    Test
    :param column: Column from DataFrame
    :return:
    """
    dates = []
    for i in range(len(column.values)):
        dtime = float(column.values.item(i))
        dates.append(strftime("%b %d, %Y %H:%M:%S", gmtime(dtime)))

    return pandas.Series(dates)


def scale_df(dframe):
    """
    Test
    :param dframe: DataFrame to be scaled
    :return:
    """
    scaler = StandardScaler()
    return pandas.DataFrame(scaler.fit_transform(dframe), columns=dframe.columns)

def text2binit(column):
    """
    Test
    :param column: Column from DataFrame
    :return:
    """
    bint = []
    for i in range(len(column.values)):
        binary = map(bin, bytearray(column.values.item(i)))
        out = 0
        for bit in binary:
            out += int(bit, 2)
        bint.append(out)

    return pandas.Series(bint)