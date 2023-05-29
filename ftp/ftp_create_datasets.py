import pandas

from probe.utils import text2binint

# Loading PCAP & Events Data
pcap_1127_0900 = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/processed/n_2012_1127_0900.csv")
pcap_1127_1000 = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/processed/a_2012_1127_1000.csv")
pcap_1127_1100 = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/processed/a_2012_1127_1100.csv")

pcap_1128_0855 = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/processed/a_2012_1128_0855.csv")
pcap_1128_0955 = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/processed/n_2012_1128_0955.csv")
pcap_1128_1055 = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/processed/n_2012_1128_1055.csv")

events = pandas.read_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/processed/events.csv")
events = events.drop("index", axis=1)

pcap_1127 = pcap_1127_1000.append(pcap_1127_1100, ignore_index=True)
pcap_1127 = pcap_1127.drop("index", axis=1)

pcap_1128 = pcap_1128_0855.append(pcap_1128_0955, ignore_index=True)
pcap_1128 = pcap_1128.append(pcap_1128_1055, ignore_index=True)
pcap_1128 = pcap_1128.drop("index", axis=1)


#Create Unlabeled Data
ulabeled_1127_104135 = pcap_1127[~pcap_1127['datetime'].str.match("Nov 27, 2012 10:41:35", na=False)]
ulabeled_1127_110230 = pcap_1127[~pcap_1127['datetime'].str.match("Nov 27, 2012 11:02:30", na=False)]
ulabeled = ulabeled_1127_104135.append(ulabeled_1127_110230, ignore_index=True)
ulabeled['info'] = text2binint(ulabeled['info'])
ulabeled = ulabeled[["datetime", "frame_num", "frame_length","eth_src", "eth_dst", "src_ip", "src_port",
                      "dst_ip", "dst_port", "window_size","header_length", "info"]]
ulabeled.to_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/datasets/unlabeled.csv")

#Create Labeled Data
attack_1127_104135 = pcap_1127[pcap_1127['datetime'].str.match("Nov 27, 2012 10:41:35", na=False)]
attack_1127_110230 = pcap_1127[pcap_1127['datetime'].str.match("Nov 27, 2012 11:02:30", na=False)]
attack_1127_111219 = pcap_1127[pcap_1127['datetime'].str.match("Nov 27, 2012 11:12:19", na=False)]
attack_1127_111705 = pcap_1127[pcap_1127['datetime'].str.match("Nov 27, 2012 11:17:05", na=False)]
attack_1127_113330 = pcap_1127[pcap_1127['datetime'].str.match("Nov 27, 2012 11:33:30", na=False)]
attack_1127_113800 = pcap_1127[pcap_1127['datetime'].str.match("Nov 27, 2012 11:38:00", na=False)]

attack_1128_094932 = pcap_1128[pcap_1128['datetime'].str.match("Nov 28, 2012 09:49:32", na=False)]
attack_1128_094940 = pcap_1128[pcap_1128['datetime'].str.match("Nov 28, 2012 09:49:40", na=False)]
attack_1128_094946 = pcap_1128[pcap_1128['datetime'].str.match("Nov 28, 2012 09:49:46", na=False)]
attack_1128_094948 = pcap_1128[pcap_1128['datetime'].str.match("Nov 28, 2012 09:49:48", na=False)]
attack_1128_101000 = pcap_1128[pcap_1128['datetime'].str.match("Nov 28, 2012 10:10:00", na=False)]

attacks = attack_1127_104135.append(attack_1127_110230, ignore_index=True)
attacks = attacks.append(attack_1127_111219, ignore_index=True)
attacks = attacks.append(attack_1127_111705, ignore_index=True)
attacks = attacks.append(attack_1127_113330, ignore_index=True)
attacks = attacks.append(attack_1127_113800, ignore_index=True)
attacks = attacks.append(attack_1128_094932, ignore_index=True)
attacks = attacks.append(attack_1128_094940, ignore_index=True)
attacks = attacks.append(attack_1128_094946, ignore_index=True)
attacks = attacks.append(attack_1128_094948, ignore_index=True)
attacks = attacks.append(attack_1128_101000, ignore_index=True)
attacks['attack'] = 1 # attack

normal = pcap_1127_0900.iloc[:1598,:]
normal['attack'] = 0 # normal

labeled = attacks.append(normal, ignore_index=True)
labeled = labeled[["datetime", "frame_num", "frame_length","eth_src", "eth_dst", "src_ip", "src_port",
                   "dst_ip", "dst_port", "window_size","header_length", "info", "attack"]]
labeled['info'] = text2binint(labeled['info'])
labeled.to_csv("/Users/patrickday/Research/autonomic-python-probe/resources/ftp_bgskrot_ex/datasets/labeled.csv")

