# read the last field of the kddcup data set - form a bar graph of the normal vs attack cases

import csv
import matplotlib.pyplot as plt

num_normal = 0
num_attack = 0 # treating everything that is not normal as attack
num_smurf = 0
num_rootkit = 0
num_spy = 0
num_warezclient = 0
num_warezmaster = 0
num_multihop = 0
num_phf = 0
num_poseidon = 0
num_nmap = 0
num_imap = 0
num_ftp_write = 0
num_satan = 0
num_pod = 0
num_bufferoverflow = 0
num_loadmodule = 0
num_perl = 0
num_guess_passwd = 0
num_teardrop = 0
num_portsweep = 0
num_ipsweep = 0
num_land = 0

file = open('kddcup.data.txt')

csvreader = csv.reader(file)
for row in csvreader:
    if (row[41] == 'normal.'):
        num_normal += 1
    elif (row[41] == 'smurf.'):
        num_smurf += 1
    elif (row[41] == 'rootkit.'):
        num_rootkit += 1
    elif (row[41] == 'spy.'):
        num_spy += 1
    elif (row[41] == 'warezclient.'):
        num_warezclient += 1
    elif (row[41] == 'warezmaster.'):
        num_warezmaster += 1
    elif (row[41] == 'multihop.'):
        num_multihop += 1
    elif (row[41] == 'phf.'):
        num_phf += 1
    elif (row[41] == 'poseidon.'):
        num_poseidon += 1
    elif (row[41] == 'nmap.'):
        num_nmap += 1
    elif (row[41] == 'imap.'):
        num_imap  += 1
    elif (row[41] == 'ftp_write.'):
        num_ftp_write += 1
    elif (row[41] == 'satan.'):
        num_satan += 1
    elif (row[41] == 'pod.'):
        num_pod += 1
    elif (row[41] == 'bufferoverflow.'):
        num_bufferoverflow += 1
    elif (row[41] == 'loadmodule.'):
        num_loadmodule += 1
    elif (row[41] == 'perl.'):
        num_perl += 1
    elif (row[41] == 'guess_passwd.'):
        num_guess_passwd += 1
    elif (row[41] == 'teardrop.'):
        num_teardrop += 1
    elif (row[41] == 'portsweep.'):
        num_portsweep += 1
    elif (row[41] == 'ipsweep.'):
        num_ipsweep += 1
    elif (row[41] == 'land.'):
        num_land += 1

labels = ['num_normal', 'num_attack', 'num_smurf', 'num_rootkit', 'num_spy', 'num_warezclient', 'num_warezmaster', 'num_multihop', 'num_phf', 'num_poseidon', 'num_nmap', 'num_imap', 'num_ftp_write', 'num_satan', 'num_pod', 'num_bufferoverflow', 'num_loadmodule', 'num_perl', 'num_guess_passwd', 'num_teardrop', 'num_portsweep', 'num_ipsweep', 'num_land']
label_vals = [num_normal, num_attack, num_smurf, num_rootkit, num_spy, num_warezclient, num_warezmaster, num_multihop, num_phf, num_poseidon, num_nmap, num_imap, num_ftp_write, num_satan, num_pod, num_bufferoverflow, num_loadmodule, num_perl, num_guess_passwd, num_teardrop, num_portsweep, num_ipsweep, num_land]
    
num_attack = num_smurf + num_rootkit + num_spy + num_warezclient + num_warezmaster + num_multihop + num_phf + num_poseidon + num_nmap + num_imap + num_ftp_write + num_satan + num_pod + num_bufferoverflow + num_loadmodule + num_perl + num_guess_passwd + num_teardrop + num_portsweep + num_ipsweep + num_land
total = num_normal + num_attack

# plot normal vs attack
#fields = ['normal', 'attack']
#field_vals = [num_normal, num_attack]

# plot all attack types
fields = labels
field_vals = label_vals

plt.bar(fields, field_vals)
plt.title('Data Labeled Normal vs Attack', fontsize=14)
plt.suptitle('Total Normal: ' + str(num_normal) + ' Total Attack: ' + str(num_attack) + ' Total: ' + str(total), fontsize=10)
plt.xlabel('Classification', fontsize=14)
plt.ylabel('Number Classified', fontsize=14)
plt.grid(True)
plt.show()
# plt.savefig('barplot_dataset_condensed.png') # only normal vs attack
plt.savefig('barplot_dataset_explanded.png') # includes all attack types
