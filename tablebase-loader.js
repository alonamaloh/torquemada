/**
 * Tablebase loader using OPFS (Origin Private File System)
 * Handles downloading, storing, and loading DTM tablebases
 */

// Base URL for tablebase files (configure as needed)
const TABLEBASE_BASE_URL = './tablebases/';

// List of DTM tablebase files (up to 5 pieces)
const DTM_FILES = [
    'dtm_000011.bin', 'dtm_000012.bin', 'dtm_000013.bin', 'dtm_000014.bin',
    'dtm_000021.bin', 'dtm_000022.bin', 'dtm_000023.bin', 'dtm_000031.bin',
    'dtm_000032.bin', 'dtm_000041.bin', 'dtm_000110.bin', 'dtm_000111.bin',
    'dtm_000112.bin', 'dtm_000113.bin', 'dtm_000120.bin', 'dtm_000121.bin',
    'dtm_000122.bin', 'dtm_000130.bin', 'dtm_000131.bin', 'dtm_000140.bin',
    'dtm_000210.bin', 'dtm_000211.bin', 'dtm_000212.bin', 'dtm_000220.bin',
    'dtm_000221.bin', 'dtm_000230.bin', 'dtm_000310.bin', 'dtm_000311.bin',
    'dtm_000320.bin', 'dtm_000410.bin', 'dtm_001001.bin', 'dtm_001002.bin',
    'dtm_001003.bin', 'dtm_001004.bin', 'dtm_001011.bin', 'dtm_001012.bin',
    'dtm_001013.bin', 'dtm_001021.bin', 'dtm_001022.bin', 'dtm_001031.bin',
    'dtm_001100.bin', 'dtm_001101.bin', 'dtm_001102.bin', 'dtm_001103.bin',
    'dtm_001110.bin', 'dtm_001111.bin', 'dtm_001112.bin', 'dtm_001120.bin',
    'dtm_001121.bin', 'dtm_001130.bin', 'dtm_001200.bin', 'dtm_001201.bin',
    'dtm_001202.bin', 'dtm_001210.bin', 'dtm_001211.bin', 'dtm_001220.bin',
    'dtm_001300.bin', 'dtm_001301.bin', 'dtm_001310.bin', 'dtm_001400.bin',
    'dtm_002001.bin', 'dtm_002002.bin', 'dtm_002003.bin', 'dtm_002011.bin',
    'dtm_002012.bin', 'dtm_002021.bin', 'dtm_002100.bin', 'dtm_002101.bin',
    'dtm_002102.bin', 'dtm_002110.bin', 'dtm_002111.bin', 'dtm_002120.bin',
    'dtm_002200.bin', 'dtm_002201.bin', 'dtm_002210.bin', 'dtm_002300.bin',
    'dtm_003001.bin', 'dtm_003002.bin', 'dtm_003011.bin', 'dtm_003100.bin',
    'dtm_003101.bin', 'dtm_003110.bin', 'dtm_003200.bin', 'dtm_004001.bin',
    'dtm_004100.bin', 'dtm_010010.bin', 'dtm_010011.bin', 'dtm_010012.bin',
    'dtm_010013.bin', 'dtm_010020.bin', 'dtm_010021.bin', 'dtm_010022.bin',
    'dtm_010030.bin', 'dtm_010031.bin', 'dtm_010040.bin', 'dtm_010110.bin',
    'dtm_010111.bin', 'dtm_010112.bin', 'dtm_010120.bin', 'dtm_010121.bin',
    'dtm_010130.bin', 'dtm_010210.bin', 'dtm_010211.bin', 'dtm_010220.bin',
    'dtm_010310.bin', 'dtm_011000.bin', 'dtm_011001.bin', 'dtm_011002.bin',
    'dtm_011003.bin', 'dtm_011010.bin', 'dtm_011011.bin', 'dtm_011012.bin',
    'dtm_011020.bin', 'dtm_011021.bin', 'dtm_011030.bin', 'dtm_011100.bin',
    'dtm_011101.bin', 'dtm_011102.bin', 'dtm_011110.bin', 'dtm_011111.bin',
    'dtm_011120.bin', 'dtm_011200.bin', 'dtm_011201.bin', 'dtm_011210.bin',
    'dtm_011300.bin', 'dtm_012000.bin', 'dtm_012001.bin', 'dtm_012002.bin',
    'dtm_012010.bin', 'dtm_012011.bin', 'dtm_012020.bin', 'dtm_012100.bin',
    'dtm_012101.bin', 'dtm_012110.bin', 'dtm_012200.bin', 'dtm_013000.bin',
    'dtm_013001.bin', 'dtm_013010.bin', 'dtm_013100.bin', 'dtm_014000.bin',
    'dtm_020010.bin', 'dtm_020011.bin', 'dtm_020012.bin', 'dtm_020020.bin',
    'dtm_020021.bin', 'dtm_020030.bin', 'dtm_020110.bin', 'dtm_020111.bin',
    'dtm_020120.bin', 'dtm_020210.bin', 'dtm_021000.bin', 'dtm_021001.bin',
    'dtm_021002.bin', 'dtm_021010.bin', 'dtm_021011.bin', 'dtm_021020.bin',
    'dtm_021100.bin', 'dtm_021101.bin', 'dtm_021110.bin', 'dtm_021200.bin',
    'dtm_022000.bin', 'dtm_022001.bin', 'dtm_022010.bin', 'dtm_022100.bin',
    'dtm_023000.bin', 'dtm_030010.bin', 'dtm_030011.bin', 'dtm_030020.bin',
    'dtm_030110.bin', 'dtm_031000.bin', 'dtm_031001.bin', 'dtm_031010.bin',
    'dtm_031100.bin', 'dtm_032000.bin', 'dtm_040010.bin', 'dtm_041000.bin',
    'dtm_100001.bin', 'dtm_100002.bin', 'dtm_100003.bin', 'dtm_100004.bin',
    'dtm_100011.bin', 'dtm_100012.bin', 'dtm_100013.bin', 'dtm_100021.bin',
    'dtm_100022.bin', 'dtm_100031.bin', 'dtm_100100.bin', 'dtm_100101.bin',
    'dtm_100102.bin', 'dtm_100103.bin', 'dtm_100110.bin', 'dtm_100111.bin',
    'dtm_100112.bin', 'dtm_100120.bin', 'dtm_100121.bin', 'dtm_100130.bin',
    'dtm_100200.bin', 'dtm_100201.bin', 'dtm_100202.bin', 'dtm_100210.bin',
    'dtm_100211.bin', 'dtm_100220.bin', 'dtm_100300.bin', 'dtm_100301.bin',
    'dtm_100310.bin', 'dtm_100400.bin', 'dtm_101001.bin', 'dtm_101002.bin',
    'dtm_101003.bin', 'dtm_101011.bin', 'dtm_101012.bin', 'dtm_101021.bin',
    'dtm_101100.bin', 'dtm_101101.bin', 'dtm_101102.bin', 'dtm_101110.bin',
    'dtm_101111.bin', 'dtm_101120.bin', 'dtm_101200.bin', 'dtm_101201.bin',
    'dtm_101210.bin', 'dtm_101300.bin', 'dtm_102001.bin', 'dtm_102002.bin',
    'dtm_102011.bin', 'dtm_102100.bin', 'dtm_102101.bin', 'dtm_102110.bin',
    'dtm_102200.bin', 'dtm_103001.bin', 'dtm_103100.bin', 'dtm_110000.bin',
    'dtm_110001.bin', 'dtm_110002.bin', 'dtm_110003.bin', 'dtm_110010.bin',
    'dtm_110011.bin', 'dtm_110012.bin', 'dtm_110020.bin', 'dtm_110021.bin',
    'dtm_110030.bin', 'dtm_110100.bin', 'dtm_110101.bin', 'dtm_110102.bin',
    'dtm_110110.bin', 'dtm_110111.bin', 'dtm_110120.bin', 'dtm_110200.bin',
    'dtm_110201.bin', 'dtm_110210.bin', 'dtm_110300.bin', 'dtm_111000.bin',
    'dtm_111001.bin', 'dtm_111002.bin', 'dtm_111010.bin', 'dtm_111011.bin',
    'dtm_111020.bin', 'dtm_111100.bin', 'dtm_111101.bin', 'dtm_111110.bin',
    'dtm_111200.bin', 'dtm_112000.bin', 'dtm_112001.bin', 'dtm_112010.bin',
    'dtm_112100.bin', 'dtm_113000.bin', 'dtm_120000.bin', 'dtm_120001.bin',
    'dtm_120002.bin', 'dtm_120010.bin', 'dtm_120011.bin', 'dtm_120020.bin',
    'dtm_120100.bin', 'dtm_120101.bin', 'dtm_120110.bin', 'dtm_120200.bin',
    'dtm_121000.bin', 'dtm_121001.bin', 'dtm_121010.bin', 'dtm_121100.bin',
    'dtm_122000.bin', 'dtm_130000.bin', 'dtm_130001.bin', 'dtm_130010.bin',
    'dtm_130100.bin', 'dtm_131000.bin', 'dtm_140000.bin', 'dtm_200001.bin',
    'dtm_200002.bin', 'dtm_200003.bin', 'dtm_200011.bin', 'dtm_200012.bin',
    'dtm_200021.bin', 'dtm_200100.bin', 'dtm_200101.bin', 'dtm_200102.bin',
    'dtm_200110.bin', 'dtm_200111.bin', 'dtm_200120.bin', 'dtm_200200.bin',
    'dtm_200201.bin', 'dtm_200210.bin', 'dtm_200300.bin', 'dtm_201001.bin',
    'dtm_201002.bin', 'dtm_201011.bin', 'dtm_201100.bin', 'dtm_201101.bin',
    'dtm_201110.bin', 'dtm_201200.bin', 'dtm_202001.bin', 'dtm_202100.bin',
    'dtm_210000.bin', 'dtm_210001.bin', 'dtm_210002.bin', 'dtm_210010.bin',
    'dtm_210011.bin', 'dtm_210020.bin', 'dtm_210100.bin', 'dtm_210101.bin',
    'dtm_210110.bin', 'dtm_210200.bin', 'dtm_211000.bin', 'dtm_211001.bin',
    'dtm_211010.bin', 'dtm_211100.bin', 'dtm_212000.bin', 'dtm_220000.bin',
    'dtm_220001.bin', 'dtm_220010.bin', 'dtm_220100.bin', 'dtm_221000.bin',
    'dtm_230000.bin', 'dtm_300001.bin', 'dtm_300002.bin', 'dtm_300011.bin',
    'dtm_300100.bin', 'dtm_300101.bin', 'dtm_300110.bin', 'dtm_300200.bin',
    'dtm_301001.bin', 'dtm_301100.bin', 'dtm_310000.bin', 'dtm_310001.bin',
    'dtm_310010.bin', 'dtm_310100.bin', 'dtm_311000.bin', 'dtm_320000.bin',
    'dtm_400001.bin', 'dtm_400100.bin', 'dtm_410000.bin',
];

// List of CWDL (compressed WDL) tablebase files (6-7 pieces, conjugate-selected)
const CWDL_FILES = [
    'cwdl_000011.bin', 'cwdl_000012.bin', 'cwdl_000013.bin', 'cwdl_000014.bin',
    'cwdl_000015.bin', 'cwdl_000022.bin', 'cwdl_000023.bin', 'cwdl_000033.bin',
    'cwdl_000034.bin', 'cwdl_000042.bin', 'cwdl_000052.bin', 'cwdl_000061.bin',
    'cwdl_000110.bin', 'cwdl_000111.bin', 'cwdl_000112.bin', 'cwdl_000113.bin',
    'cwdl_000114.bin', 'cwdl_000120.bin', 'cwdl_000122.bin', 'cwdl_000130.bin',
    'cwdl_000131.bin', 'cwdl_000133.bin', 'cwdl_000141.bin', 'cwdl_000142.bin',
    'cwdl_000151.bin', 'cwdl_000211.bin', 'cwdl_000212.bin', 'cwdl_000213.bin',
    'cwdl_000220.bin', 'cwdl_000222.bin', 'cwdl_000230.bin', 'cwdl_000231.bin',
    'cwdl_000232.bin', 'cwdl_000240.bin', 'cwdl_000241.bin', 'cwdl_000250.bin',
    'cwdl_000310.bin', 'cwdl_000311.bin', 'cwdl_000312.bin', 'cwdl_000320.bin',
    'cwdl_000321.bin', 'cwdl_000330.bin', 'cwdl_000331.bin', 'cwdl_000340.bin',
    'cwdl_000410.bin', 'cwdl_000411.bin', 'cwdl_000412.bin', 'cwdl_000420.bin',
    'cwdl_000421.bin', 'cwdl_000430.bin', 'cwdl_000510.bin', 'cwdl_000511.bin',
    'cwdl_000520.bin', 'cwdl_000610.bin', 'cwdl_001004.bin', 'cwdl_001005.bin',
    'cwdl_001006.bin', 'cwdl_001012.bin', 'cwdl_001023.bin', 'cwdl_001032.bin',
    'cwdl_001042.bin', 'cwdl_001051.bin', 'cwdl_001100.bin', 'cwdl_001104.bin',
    'cwdl_001105.bin', 'cwdl_001110.bin', 'cwdl_001111.bin', 'cwdl_001120.bin',
    'cwdl_001121.bin', 'cwdl_001122.bin', 'cwdl_001130.bin', 'cwdl_001131.bin',
    'cwdl_001132.bin', 'cwdl_001141.bin', 'cwdl_001200.bin', 'cwdl_001204.bin',
    'cwdl_001211.bin', 'cwdl_001220.bin', 'cwdl_001221.bin', 'cwdl_001222.bin',
    'cwdl_001230.bin', 'cwdl_001231.bin', 'cwdl_001240.bin', 'cwdl_001300.bin',
    'cwdl_001303.bin', 'cwdl_001310.bin', 'cwdl_001320.bin', 'cwdl_001321.bin',
    'cwdl_001330.bin', 'cwdl_001402.bin', 'cwdl_001410.bin', 'cwdl_001420.bin',
    'cwdl_001510.bin', 'cwdl_002001.bin', 'cwdl_002012.bin', 'cwdl_002032.bin',
    'cwdl_002041.bin', 'cwdl_002101.bin', 'cwdl_002110.bin', 'cwdl_002120.bin',
    'cwdl_002121.bin', 'cwdl_002130.bin', 'cwdl_002131.bin', 'cwdl_002200.bin',
    'cwdl_002210.bin', 'cwdl_002211.bin', 'cwdl_002220.bin', 'cwdl_002221.bin',
    'cwdl_002230.bin', 'cwdl_002300.bin', 'cwdl_002310.bin', 'cwdl_002320.bin',
    'cwdl_002410.bin', 'cwdl_003022.bin', 'cwdl_003031.bin', 'cwdl_003110.bin',
    'cwdl_003111.bin', 'cwdl_003120.bin', 'cwdl_003121.bin', 'cwdl_003210.bin',
    'cwdl_003211.bin', 'cwdl_003220.bin', 'cwdl_003300.bin', 'cwdl_003310.bin',
    'cwdl_004100.bin', 'cwdl_004110.bin', 'cwdl_004111.bin', 'cwdl_004200.bin',
    'cwdl_004210.bin', 'cwdl_004300.bin', 'cwdl_005100.bin', 'cwdl_005110.bin',
    'cwdl_005200.bin', 'cwdl_006100.bin', 'cwdl_010010.bin', 'cwdl_010011.bin',
    'cwdl_010013.bin', 'cwdl_010020.bin', 'cwdl_010030.bin', 'cwdl_010031.bin',
    'cwdl_010040.bin', 'cwdl_010041.bin', 'cwdl_010042.bin', 'cwdl_010050.bin',
    'cwdl_010051.bin', 'cwdl_010060.bin', 'cwdl_010111.bin', 'cwdl_010112.bin',
    'cwdl_010113.bin', 'cwdl_010120.bin', 'cwdl_010130.bin', 'cwdl_010131.bin',
    'cwdl_010132.bin', 'cwdl_010140.bin', 'cwdl_010141.bin', 'cwdl_010150.bin',
    'cwdl_010212.bin', 'cwdl_010220.bin', 'cwdl_010221.bin', 'cwdl_010230.bin',
    'cwdl_010231.bin', 'cwdl_010240.bin', 'cwdl_010311.bin', 'cwdl_010320.bin',
    'cwdl_010321.bin', 'cwdl_010330.bin', 'cwdl_010410.bin', 'cwdl_010411.bin',
    'cwdl_010420.bin', 'cwdl_010510.bin', 'cwdl_011000.bin', 'cwdl_011003.bin',
    'cwdl_011004.bin', 'cwdl_011005.bin', 'cwdl_011020.bin', 'cwdl_011021.bin',
    'cwdl_011030.bin', 'cwdl_011031.bin', 'cwdl_011040.bin', 'cwdl_011041.bin',
    'cwdl_011050.bin', 'cwdl_011100.bin', 'cwdl_011103.bin', 'cwdl_011104.bin',
    'cwdl_011110.bin', 'cwdl_011111.bin', 'cwdl_011120.bin', 'cwdl_011130.bin',
    'cwdl_011131.bin', 'cwdl_011140.bin', 'cwdl_011200.bin', 'cwdl_011202.bin',
    'cwdl_011203.bin', 'cwdl_011210.bin', 'cwdl_011220.bin', 'cwdl_011230.bin',
    'cwdl_011302.bin', 'cwdl_011310.bin', 'cwdl_011320.bin', 'cwdl_011401.bin',
    'cwdl_012000.bin', 'cwdl_012001.bin', 'cwdl_012020.bin', 'cwdl_012021.bin',
    'cwdl_012030.bin', 'cwdl_012031.bin', 'cwdl_012040.bin', 'cwdl_012100.bin',
    'cwdl_012110.bin', 'cwdl_012120.bin', 'cwdl_012121.bin', 'cwdl_012130.bin',
    'cwdl_012200.bin', 'cwdl_012210.bin', 'cwdl_012220.bin', 'cwdl_012310.bin',
    'cwdl_013000.bin', 'cwdl_013010.bin', 'cwdl_013011.bin', 'cwdl_013020.bin',
    'cwdl_013021.bin', 'cwdl_013030.bin', 'cwdl_013100.bin', 'cwdl_013110.bin',
    'cwdl_013120.bin', 'cwdl_013200.bin', 'cwdl_013210.bin', 'cwdl_013300.bin',
    'cwdl_014000.bin', 'cwdl_014010.bin', 'cwdl_014020.bin', 'cwdl_014100.bin',
    'cwdl_014110.bin', 'cwdl_015000.bin', 'cwdl_015010.bin', 'cwdl_015100.bin',
    'cwdl_016000.bin', 'cwdl_020010.bin', 'cwdl_020011.bin', 'cwdl_020012.bin',
    'cwdl_020013.bin', 'cwdl_020014.bin', 'cwdl_020020.bin', 'cwdl_020030.bin',
    'cwdl_020031.bin', 'cwdl_020040.bin', 'cwdl_020041.bin', 'cwdl_020050.bin',
    'cwdl_020120.bin', 'cwdl_020121.bin', 'cwdl_020130.bin', 'cwdl_020131.bin',
    'cwdl_020140.bin', 'cwdl_020210.bin', 'cwdl_020211.bin', 'cwdl_020220.bin',
    'cwdl_020230.bin', 'cwdl_020310.bin', 'cwdl_020320.bin', 'cwdl_021000.bin',
    'cwdl_021003.bin', 'cwdl_021004.bin', 'cwdl_021010.bin', 'cwdl_021011.bin',
    'cwdl_021020.bin', 'cwdl_021030.bin', 'cwdl_021031.bin', 'cwdl_021040.bin',
    'cwdl_021100.bin', 'cwdl_021102.bin', 'cwdl_021103.bin', 'cwdl_021110.bin',
    'cwdl_021120.bin', 'cwdl_021130.bin', 'cwdl_021202.bin', 'cwdl_021220.bin',
    'cwdl_021301.bin', 'cwdl_021400.bin', 'cwdl_022000.bin', 'cwdl_022010.bin',
    'cwdl_022020.bin', 'cwdl_022021.bin', 'cwdl_022030.bin', 'cwdl_022100.bin',
    'cwdl_022110.bin', 'cwdl_022120.bin', 'cwdl_022210.bin', 'cwdl_023000.bin',
    'cwdl_023010.bin', 'cwdl_023020.bin', 'cwdl_023100.bin', 'cwdl_023110.bin',
    'cwdl_023200.bin', 'cwdl_024010.bin', 'cwdl_024100.bin', 'cwdl_025000.bin',
    'cwdl_030010.bin', 'cwdl_030011.bin', 'cwdl_030012.bin', 'cwdl_030013.bin',
    'cwdl_030020.bin', 'cwdl_030021.bin', 'cwdl_030030.bin', 'cwdl_030040.bin',
    'cwdl_030111.bin', 'cwdl_030120.bin', 'cwdl_030130.bin', 'cwdl_030220.bin',
    'cwdl_031001.bin', 'cwdl_031002.bin', 'cwdl_031003.bin', 'cwdl_031010.bin',
    'cwdl_031020.bin', 'cwdl_031030.bin', 'cwdl_031100.bin', 'cwdl_031101.bin',
    'cwdl_031102.bin', 'cwdl_031110.bin', 'cwdl_031120.bin', 'cwdl_031200.bin',
    'cwdl_031201.bin', 'cwdl_031300.bin', 'cwdl_032000.bin', 'cwdl_032010.bin',
    'cwdl_032020.bin', 'cwdl_032110.bin', 'cwdl_033000.bin', 'cwdl_033010.bin',
    'cwdl_033100.bin', 'cwdl_040010.bin', 'cwdl_040011.bin', 'cwdl_040012.bin',
    'cwdl_040020.bin', 'cwdl_040030.bin', 'cwdl_040110.bin', 'cwdl_040120.bin',
    'cwdl_040210.bin', 'cwdl_041000.bin', 'cwdl_041001.bin', 'cwdl_041002.bin',
    'cwdl_041010.bin', 'cwdl_041100.bin', 'cwdl_041101.bin', 'cwdl_041200.bin',
    'cwdl_042000.bin', 'cwdl_042001.bin', 'cwdl_042010.bin', 'cwdl_042100.bin',
    'cwdl_100012.bin', 'cwdl_100021.bin', 'cwdl_100022.bin', 'cwdl_100023.bin',
    'cwdl_100032.bin', 'cwdl_100033.bin', 'cwdl_100041.bin', 'cwdl_100042.bin',
    'cwdl_100051.bin', 'cwdl_100101.bin', 'cwdl_100110.bin', 'cwdl_100111.bin',
    'cwdl_100120.bin', 'cwdl_100121.bin', 'cwdl_100122.bin', 'cwdl_100123.bin',
    'cwdl_100131.bin', 'cwdl_100132.bin', 'cwdl_100141.bin', 'cwdl_100201.bin',
    'cwdl_100211.bin', 'cwdl_100220.bin', 'cwdl_100221.bin', 'cwdl_100222.bin',
    'cwdl_100230.bin', 'cwdl_100231.bin', 'cwdl_100240.bin', 'cwdl_100310.bin',
    'cwdl_100320.bin', 'cwdl_100321.bin', 'cwdl_100330.bin', 'cwdl_100410.bin',
    'cwdl_100411.bin', 'cwdl_100420.bin', 'cwdl_100510.bin', 'cwdl_101001.bin',
    'cwdl_101012.bin', 'cwdl_101022.bin', 'cwdl_101032.bin', 'cwdl_101041.bin',
    'cwdl_101110.bin', 'cwdl_101112.bin', 'cwdl_101120.bin', 'cwdl_101121.bin',
    'cwdl_101122.bin', 'cwdl_101131.bin', 'cwdl_101210.bin', 'cwdl_101211.bin',
    'cwdl_101220.bin', 'cwdl_101221.bin', 'cwdl_101230.bin', 'cwdl_101310.bin',
    'cwdl_101311.bin', 'cwdl_101320.bin', 'cwdl_101410.bin', 'cwdl_102001.bin',
    'cwdl_102011.bin', 'cwdl_102022.bin', 'cwdl_102031.bin', 'cwdl_102110.bin',
    'cwdl_102111.bin', 'cwdl_102112.bin', 'cwdl_102121.bin', 'cwdl_102210.bin',
    'cwdl_102211.bin', 'cwdl_102220.bin', 'cwdl_102310.bin', 'cwdl_102400.bin',
    'cwdl_103001.bin', 'cwdl_103021.bin', 'cwdl_103100.bin', 'cwdl_103110.bin',
    'cwdl_103111.bin', 'cwdl_103200.bin', 'cwdl_103210.bin', 'cwdl_104100.bin',
    'cwdl_104101.bin', 'cwdl_104200.bin', 'cwdl_105100.bin', 'cwdl_110000.bin',
    'cwdl_110001.bin', 'cwdl_110002.bin', 'cwdl_110003.bin', 'cwdl_110011.bin',
    'cwdl_110021.bin', 'cwdl_110022.bin', 'cwdl_110031.bin', 'cwdl_110032.bin',
    'cwdl_110040.bin', 'cwdl_110041.bin', 'cwdl_110050.bin', 'cwdl_110110.bin',
    'cwdl_110111.bin', 'cwdl_110120.bin', 'cwdl_110121.bin', 'cwdl_110122.bin',
    'cwdl_110130.bin', 'cwdl_110131.bin', 'cwdl_110140.bin', 'cwdl_110210.bin',
    'cwdl_110220.bin', 'cwdl_110221.bin', 'cwdl_110230.bin', 'cwdl_110310.bin',
    'cwdl_110320.bin', 'cwdl_110410.bin', 'cwdl_111000.bin', 'cwdl_111010.bin',
    'cwdl_111020.bin', 'cwdl_111021.bin', 'cwdl_111030.bin', 'cwdl_111031.bin',
    'cwdl_111040.bin', 'cwdl_111100.bin', 'cwdl_111110.bin', 'cwdl_111111.bin',
    'cwdl_111120.bin', 'cwdl_111121.bin', 'cwdl_111130.bin', 'cwdl_111200.bin',
    'cwdl_111210.bin', 'cwdl_111220.bin', 'cwdl_111310.bin', 'cwdl_112000.bin',
    'cwdl_112010.bin', 'cwdl_112011.bin', 'cwdl_112020.bin', 'cwdl_112021.bin',
    'cwdl_112030.bin', 'cwdl_112110.bin', 'cwdl_112111.bin', 'cwdl_112120.bin',
    'cwdl_112200.bin', 'cwdl_112210.bin', 'cwdl_113000.bin', 'cwdl_113010.bin',
    'cwdl_113011.bin', 'cwdl_113020.bin', 'cwdl_113100.bin', 'cwdl_113110.bin',
    'cwdl_113200.bin', 'cwdl_114000.bin', 'cwdl_114010.bin', 'cwdl_114100.bin',
    'cwdl_115000.bin', 'cwdl_120001.bin', 'cwdl_120002.bin', 'cwdl_120010.bin',
    'cwdl_120011.bin', 'cwdl_120020.bin', 'cwdl_120030.bin', 'cwdl_120031.bin',
    'cwdl_120040.bin', 'cwdl_120120.bin', 'cwdl_120130.bin', 'cwdl_120210.bin',
    'cwdl_120220.bin', 'cwdl_120310.bin', 'cwdl_121000.bin', 'cwdl_121003.bin',
    'cwdl_121010.bin', 'cwdl_121020.bin', 'cwdl_121021.bin', 'cwdl_121030.bin',
    'cwdl_121100.bin', 'cwdl_121110.bin', 'cwdl_121120.bin', 'cwdl_121210.bin',
    'cwdl_122000.bin', 'cwdl_122010.bin', 'cwdl_122020.bin', 'cwdl_122110.bin',
    'cwdl_122200.bin', 'cwdl_123010.bin', 'cwdl_124000.bin', 'cwdl_130000.bin',
    'cwdl_130001.bin', 'cwdl_130002.bin', 'cwdl_130010.bin', 'cwdl_130020.bin',
    'cwdl_130030.bin', 'cwdl_130120.bin', 'cwdl_131000.bin', 'cwdl_131001.bin',
    'cwdl_131002.bin', 'cwdl_131010.bin', 'cwdl_131020.bin', 'cwdl_131110.bin',
    'cwdl_132000.bin', 'cwdl_132010.bin', 'cwdl_132100.bin', 'cwdl_140000.bin',
    'cwdl_140001.bin', 'cwdl_140002.bin', 'cwdl_140010.bin', 'cwdl_140011.bin',
    'cwdl_141000.bin', 'cwdl_141001.bin', 'cwdl_141010.bin', 'cwdl_141100.bin',
    'cwdl_200012.bin', 'cwdl_200022.bin', 'cwdl_200023.bin', 'cwdl_200032.bin',
    'cwdl_200110.bin', 'cwdl_200112.bin', 'cwdl_200120.bin', 'cwdl_200121.bin',
    'cwdl_200122.bin', 'cwdl_200131.bin', 'cwdl_200210.bin', 'cwdl_200211.bin',
    'cwdl_200220.bin', 'cwdl_200221.bin', 'cwdl_200230.bin', 'cwdl_200310.bin',
    'cwdl_200311.bin', 'cwdl_200320.bin', 'cwdl_200400.bin', 'cwdl_200410.bin',
    'cwdl_201001.bin', 'cwdl_201011.bin', 'cwdl_201021.bin', 'cwdl_201022.bin',
    'cwdl_201031.bin', 'cwdl_201110.bin', 'cwdl_201111.bin', 'cwdl_201112.bin',
    'cwdl_201121.bin', 'cwdl_201210.bin', 'cwdl_201211.bin', 'cwdl_201220.bin',
    'cwdl_201310.bin', 'cwdl_202012.bin', 'cwdl_202021.bin', 'cwdl_202100.bin',
    'cwdl_202101.bin', 'cwdl_202110.bin', 'cwdl_202111.bin', 'cwdl_202200.bin',
    'cwdl_202210.bin', 'cwdl_203011.bin', 'cwdl_203100.bin', 'cwdl_203101.bin',
    'cwdl_203200.bin', 'cwdl_204001.bin', 'cwdl_210000.bin', 'cwdl_210012.bin',
    'cwdl_210021.bin', 'cwdl_210022.bin', 'cwdl_210030.bin', 'cwdl_210031.bin',
    'cwdl_210040.bin', 'cwdl_210110.bin', 'cwdl_210111.bin', 'cwdl_210120.bin',
    'cwdl_210121.bin', 'cwdl_210210.bin', 'cwdl_210211.bin', 'cwdl_210220.bin',
    'cwdl_210300.bin', 'cwdl_210310.bin', 'cwdl_211000.bin', 'cwdl_211001.bin',
    'cwdl_211010.bin', 'cwdl_211011.bin', 'cwdl_211012.bin', 'cwdl_211020.bin',
    'cwdl_211021.bin', 'cwdl_211030.bin', 'cwdl_211110.bin', 'cwdl_211111.bin',
    'cwdl_211120.bin', 'cwdl_211200.bin', 'cwdl_211210.bin', 'cwdl_211300.bin',
    'cwdl_212000.bin', 'cwdl_212010.bin', 'cwdl_212011.bin', 'cwdl_212020.bin',
    'cwdl_212100.bin', 'cwdl_212110.bin', 'cwdl_213000.bin', 'cwdl_213010.bin',
    'cwdl_213100.bin', 'cwdl_214000.bin', 'cwdl_220000.bin', 'cwdl_220001.bin',
    'cwdl_220002.bin', 'cwdl_220011.bin', 'cwdl_220021.bin', 'cwdl_220030.bin',
    'cwdl_220110.bin', 'cwdl_220111.bin', 'cwdl_220120.bin', 'cwdl_220200.bin',
    'cwdl_220210.bin', 'cwdl_221000.bin', 'cwdl_221010.bin', 'cwdl_221020.bin',
    'cwdl_221100.bin', 'cwdl_221110.bin', 'cwdl_222010.bin', 'cwdl_222100.bin',
    'cwdl_223000.bin', 'cwdl_230001.bin', 'cwdl_230002.bin', 'cwdl_230010.bin',
    'cwdl_230011.bin', 'cwdl_230020.bin', 'cwdl_230100.bin', 'cwdl_230110.bin',
    'cwdl_231000.bin', 'cwdl_231010.bin', 'cwdl_232000.bin', 'cwdl_240000.bin',
    'cwdl_240001.bin', 'cwdl_240010.bin', 'cwdl_241000.bin', 'cwdl_300013.bin',
    'cwdl_300022.bin', 'cwdl_300100.bin', 'cwdl_300111.bin', 'cwdl_300112.bin',
    'cwdl_300121.bin', 'cwdl_300210.bin', 'cwdl_300211.bin', 'cwdl_300220.bin',
    'cwdl_300310.bin', 'cwdl_300400.bin', 'cwdl_301001.bin', 'cwdl_301012.bin',
    'cwdl_301021.bin', 'cwdl_301111.bin', 'cwdl_301200.bin', 'cwdl_301210.bin',
    'cwdl_302001.bin', 'cwdl_302011.bin', 'cwdl_302101.bin', 'cwdl_302200.bin',
    'cwdl_303001.bin', 'cwdl_310011.bin', 'cwdl_310012.bin', 'cwdl_310021.bin',
    'cwdl_310030.bin', 'cwdl_310111.bin', 'cwdl_310210.bin', 'cwdl_310300.bin',
    'cwdl_311000.bin', 'cwdl_311001.bin', 'cwdl_311010.bin', 'cwdl_311011.bin',
    'cwdl_311020.bin', 'cwdl_311100.bin', 'cwdl_311110.bin', 'cwdl_312000.bin',
    'cwdl_312001.bin', 'cwdl_312010.bin', 'cwdl_312100.bin', 'cwdl_313000.bin',
    'cwdl_320000.bin', 'cwdl_320110.bin', 'cwdl_321010.bin', 'cwdl_321100.bin',
    'cwdl_322000.bin', 'cwdl_330000.bin', 'cwdl_330010.bin', 'cwdl_330100.bin',
    'cwdl_340000.bin', 'cwdl_400012.bin', 'cwdl_400102.bin', 'cwdl_400111.bin',
    'cwdl_400300.bin', 'cwdl_401011.bin', 'cwdl_401101.bin', 'cwdl_410002.bin',
    'cwdl_410200.bin', 'cwdl_411000.bin', 'cwdl_411001.bin', 'cwdl_411010.bin',
    'cwdl_412000.bin', 'cwdl_421000.bin',
];

/**
 * TablebaseLoader class - manages tablebase storage and retrieval
 */
export class TablebaseLoader {
    constructor() {
        this.opfsRoot = null;
        this.tbDirectory = null;
        this.loadedFiles = new Set();
        this.onProgress = null;
        this.isInitialized = false;
    }

    /**
     * Initialize OPFS access
     */
    async init() {
        if (!('storage' in navigator) || !('getDirectory' in navigator.storage)) {
            throw new Error('OPFS not supported in this browser');
        }

        this.opfsRoot = await navigator.storage.getDirectory();
        this.tbDirectory = await this.opfsRoot.getDirectoryHandle('tablebases', { create: true });
        this.isInitialized = true;
    }

    /**
     * Check if OPFS is available and initialized
     */
    isAvailable() {
        return this.isInitialized && this.tbDirectory !== null;
    }

    /**
     * Check which tablebases are already stored in OPFS
     */
    async checkStoredTablebases() {
        if (!this.isAvailable()) {
            return [];
        }
        const stored = [];
        for await (const entry of this.tbDirectory.values()) {
            if (entry.kind === 'file' && entry.name.startsWith('dtm_')) {
                stored.push(entry.name);
            }
        }
        return stored;
    }

    /**
     * Download missing tablebases from server
     * @param {Function} onProgress - Callback(loaded, total, currentFile)
     */
    async downloadTablebases(onProgress = null) {
        if (!this.isAvailable()) {
            throw new Error('OPFS not available. Tablebase storage requires a browser with Origin Private File System support.');
        }

        this.onProgress = onProgress;

        // Check what's already stored
        const stored = await this.checkStoredTablebases();
        const storedSet = new Set(stored);

        // Filter to only missing files
        const missing = DTM_FILES.filter(f => !storedSet.has(f));

        if (missing.length === 0) {
            if (onProgress) onProgress(DTM_FILES.length, DTM_FILES.length, 'Complete');
            return { downloaded: 0, total: DTM_FILES.length };
        }

        let downloaded = stored.length;
        const total = DTM_FILES.length;

        for (const filename of missing) {
            try {
                if (onProgress) onProgress(downloaded + 1, total, filename);

                const response = await fetch(TABLEBASE_BASE_URL + filename);
                if (!response.ok) {
                    console.warn(`Failed to download ${filename}: ${response.status}`);
                    continue;
                }

                const data = await response.arrayBuffer();

                // Store in OPFS
                const fileHandle = await this.tbDirectory.getFileHandle(filename, { create: true });
                const writable = await fileHandle.createWritable();
                await writable.write(data);
                await writable.close();

                downloaded++;
            } catch (err) {
                console.warn(`Error downloading ${filename}:`, err);
            }
        }

        if (onProgress) onProgress(downloaded, total, 'Complete');
        return { downloaded: downloaded - stored.length, total };
    }

    /**
     * Load a tablebase file from OPFS (sync access for Worker)
     * Returns Uint8Array or null if not found
     */
    async loadTablebase(filename) {
        if (!this.isAvailable()) {
            return null;
        }
        try {
            const fileHandle = await this.tbDirectory.getFileHandle(filename);
            const file = await fileHandle.getFile();
            const buffer = await file.arrayBuffer();
            return new Uint8Array(buffer);
        } catch (err) {
            return null;
        }
    }

    /**
     * Load all tablebases and return them as a map
     * @returns {Map<string, Uint8Array>}
     */
    async loadAllTablebases() {
        const tablebases = new Map();

        if (!this.isAvailable()) {
            return tablebases;
        }

        for await (const entry of this.tbDirectory.values()) {
            if (entry.kind === 'file' && entry.name.startsWith('dtm_')) {
                const data = await this.loadTablebase(entry.name);
                if (data) {
                    // Extract material key from filename (dtm_XXXXXX.bin -> XXXXXX)
                    const materialKey = entry.name.slice(4, 10);
                    tablebases.set(materialKey, data);
                }
            }
        }

        return tablebases;
    }

    /**
     * Get total size of stored tablebases
     */
    async getStoredSize() {
        if (!this.isAvailable()) {
            return 0;
        }
        let total = 0;
        for await (const entry of this.tbDirectory.values()) {
            if (entry.kind === 'file') {
                const file = await (await this.tbDirectory.getFileHandle(entry.name)).getFile();
                total += file.size;
            }
        }
        return total;
    }

    /**
     * Check which CWDL tablebases are already stored in OPFS
     */
    async checkStoredCWDLTablebases() {
        if (!this.isAvailable()) {
            return [];
        }
        const stored = [];
        for await (const entry of this.tbDirectory.values()) {
            if (entry.kind === 'file' && entry.name.startsWith('cwdl_')) {
                stored.push(entry.name);
            }
        }
        return stored;
    }

    /**
     * Download missing CWDL tablebases from server
     * @param {Function} onProgress - Callback(loaded, total, currentFile)
     */
    async downloadCWDLTablebases(onProgress = null) {
        if (!this.isAvailable()) {
            throw new Error('OPFS not available.');
        }

        // Check what's already stored
        const stored = await this.checkStoredCWDLTablebases();
        const storedSet = new Set(stored);

        // Filter to only missing files
        const missing = CWDL_FILES.filter(f => !storedSet.has(f));

        if (missing.length === 0) {
            if (onProgress) onProgress(CWDL_FILES.length, CWDL_FILES.length, 'Complete');
            return { downloaded: 0, total: CWDL_FILES.length };
        }

        let downloaded = stored.length;
        const total = CWDL_FILES.length;

        for (const filename of missing) {
            try {
                if (onProgress) onProgress(downloaded + 1, total, filename);

                const response = await fetch(TABLEBASE_BASE_URL + filename);
                if (!response.ok) {
                    console.warn(`Failed to download ${filename}: ${response.status}`);
                    continue;
                }

                const data = await response.arrayBuffer();

                // Store in OPFS
                const fileHandle = await this.tbDirectory.getFileHandle(filename, { create: true });
                const writable = await fileHandle.createWritable();
                await writable.write(data);
                await writable.close();

                downloaded++;
            } catch (err) {
                console.warn(`Error downloading ${filename}:`, err);
            }
        }

        if (onProgress) onProgress(downloaded, total, 'Complete');
        return { downloaded: downloaded - stored.length, total };
    }

    /**
     * Clear all stored tablebases
     */
    async clearTablebases() {
        if (!this.isAvailable()) {
            return;
        }
        for await (const entry of this.tbDirectory.values()) {
            if (entry.kind === 'file') {
                await this.tbDirectory.removeEntry(entry.name);
            }
        }
    }
}

/**
 * Load NN model file from server
 */
export async function loadNNModelFile(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load model: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
}

export { DTM_FILES, CWDL_FILES };
