Pcap_dir = "1_Pcap/"
csv_dir  = "2_Csv/"
Weight_dir = "4_Weight/"
result_dir = "5_Result/"
analysis_dir = "6_Analysis/"

train_joy_csv_filename = "train_joy.csv"
test_joy_csv_filename  = "test_joy.csv"
train_handcraft_csv_filename = "train_handcraft.csv"
test_handcraft_csv_filename  = "test_handcraft.csv"
zeroshot_joy_csv_filename = "zeroshot_joy.csv"
zeroshot_handcraft_csv_filename = "zeroshot_handcraft.csv"

train_csv_joy = csv_dir + train_joy_csv_filename
test_csv_joy  = csv_dir + test_joy_csv_filename
train_csv_handcraft = csv_dir + train_handcraft_csv_filename
test_csv_handcraft  = csv_dir + test_handcraft_csv_filename
zeroshot_csv_joy = csv_dir + zeroshot_joy_csv_filename
zeroshot_csv_handcraft = csv_dir + zeroshot_handcraft_csv_filename

joy_path = "../joy/bin/joy"

class_num = 11
class_dir = ["EITest/", "Emotet/", "Hancitor/", "Nuclear/", "Rig/", "TrickBot/", "Dridex/", "Razy/", "HTBot/", "wannacry/", "normal/"]
labels = ["EITest", "Emotet", "Hancitor", "Nuclear", "Rig", "TrickBot", "Dridex", "Razy", "HTBot", "wannacry", "normal"]
dic = {"EITest": 0, "Emotet": 1, "Hancitor": 2, "Nuclear": 3, "Rig": 4, "TrickBot": 5, "Dridex": 6, "Razy": 7, "HTBot":8, "wannacry":9, "normal":10}
normal_label = dic["normal"]

zeroshot_class_dir = ["Shade/", "Cerber/", "ctu17/", "ctu29/", "ctu54/", "ctu74/", "ctu82/", "ctu83/", "ctu97/", "ctu103/"]
zeroshot_labels = ["Shade", "Cerber", "ctu17", "ctu29", "ctu54", "ctu74", "ctu82", "ctu83", "ctu97", "ctu103"]

binary_labels = ["normal", "malicious"]

# for feature analysis
bin_range = 500
