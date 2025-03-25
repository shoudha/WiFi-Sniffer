Abstract. In this project, IEEE 802.11g Wi-Fi packets, captured using NI USRP and LabVIEW, are demodulated in MATLAB to extract information of the signal.
Background. IEEE 802.11g Wi-Fi uses OFDM (Orthogonal Frequency Division Multiplexing) and operates in frequency band 2.400-2.4835 GHz. The entire band is divided into 14 channels, spaced 5 MHz apart from each other except for a 12 MHz space before channel 14. Each channel is 16.25 MHz wide and is divided into 64 subcarriers with 12 null (one DC), 48 data and 4 pilot subcarriers. The pilot subcarriers are located at subcarriers -21, -7, 7 and 21. The sampling rate is 20MHz and only channel 1 is used in this project due to Wi-Fi signal availability in that channel.
802.11g Wi-Fi packets start with a preamble field of 16µs, consisting of short and long preamble fields of 8µs each. There are 10 repeated fields, each of 0.8µs, in the short preamble field. This repetitive pattern is used for frame synchronization and carrier frequency offset compensation (CFO). The long preamble field uses OFDM and consists of two repeated fields of 3.2µs each and a guard interval of 1.6µs, which is a cyclic prefix of the long preamble field. The long preamble is used for channel estimation. The pilot subcarriers are used for residual frequency (RF) compensation.
The SIGNAL field is 4µs, with a guard interval (cyclic prefix) of 0.8µs in the beginning. It is BPSK modulated and contains the data rate, length information of the Wi-Fi packet. From these information, other information for example: modulation scheme, convolutional coding rate, coded bits and data bits per sub carrier etc. can be extracted from the 802.11 IEEE standard [1]. Rate depended variables’ table is presented below:
Rate	Data Rate
(Mbps)	Modulation	Coding Rate
(R)	Coded Bits per Subcarrier
(NBPSC)	Coded Bits per OFDM Symbol
(NCBPS)	Data Bits per OFDM Symbol
(NDBPS)
1101	6	BPSK	1/2	1	48	24
1111	9	BPSK	3/4	1	48	36
0101	12	QPSK	1/2	2	96	48
0111	18	QPSK	3/4	2	96	72
1001	24	16-QAM	1/2	4	192	96
1011	36	16-QAM	3/4	4	192	144
0001	48	64-QAM	2/3	6	288	192
0011	54	64-QAM	3/4	6	288	216
Table 1. Rate depended variables for IEEE 802.11g Wi-Fi packet
