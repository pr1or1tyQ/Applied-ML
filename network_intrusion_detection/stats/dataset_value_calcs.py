'''
- what is getting misclassified in the model - model.predict
    if expected output doesn't match what it should be - save what is being misclassified by the model
- calculate false positive, false negative, true positive, true negative (comparing predicition with what it should)
    combine all attacks into 1 to make it easier
    true positive rate for attack (is attack and predict it as an attack)
- focus on calculating for attacks - attack fpr, tpr, fnr, and tnr (flase and true negative)

is this model accurate and actually usable (like in an ops center)
goal: high true posotive,  hight ture negative, low false positive, low false negative

if argmax reutrns a column that isn't 0 then you know it is classified as attack - use argmax to see what index was predicted - if index retuned is not the same have a missclassification *'''

import numpy as np

# open the test data and traning data and calculate the number of normal and attack outputs
# normal outputs are at the 0 index position
# attack outputs are at all other index positions

def calc_test_data():
    with open ('test_data.csv', 'r') as test_file:
        test_data = test_file.readlines()
        # use argmax to find the index position with the max value in the array
        # if the index position is 0 then it is a normal output
        # if the index position is not 0 then it is an attack output
        test_normal = 0
        test_attack = 0
        for i in test_data:
            i = i.strip()
            i = i.split(',')
            i = np.array(i)
            #i = i.astype(np.float)
            # print(i.shape)
            # print(type(i))
            # print(f'argmax: {i.argmax()}')
            if i.argmax() == 0:
                test_normal += 1
            else:
                test_attack += 1
        print(f'test_normal: {test_normal} \tPrecentage: {test_normal / (test_normal + test_attack)}')
        print(f'test_attack: {test_attack} \tPercentage: {test_attack / (test_normal + test_attack)}')
        print(f'test_total: {test_normal + test_attack}')
        print('test data complete')

def calc_test_labesl():
    with open ('test_labels.csv', 'r') as test_file:
        test_data = test_file.readlines()
        # use argmax to find the index position with the max value in the array
        # if the index position is 0 then it is a normal output
        # if the index position is not 0 then it is an attack output
        test_normal = 0
        test_attack = 0
        for i in test_data:
            i = i.strip()
            i = i.split(',')
            i = np.array(i)
            #i = i.astype(np.float)
            # print(i.shape)
            # print(type(i))
            # print(f'argmax: {i.argmax()}')
            if i.argmax() == 0:
                test_normal += 1
            else:
                test_attack += 1
        print(f'test_normal: {test_normal} \tPrecentage: {test_normal / (test_normal + test_attack)}')
        print(f'test_attack: {test_attack} \tPercentage: {test_attack / (test_normal + test_attack)}')
        print(f'test_total: {test_normal + test_attack}')
        print('test data complete')

def calc_train_data():
    with open ('train_data.csv', 'r') as train_file:
        train_data = train_file.readlines()
        # use argmax to find the index position with the max value in the array
        # if the index position is 0 then it is a normal output
        # if the index position is not 0 then it is an attack output
        train_normal = 0
        train_attack = 0
        for i in train_data:
            i = i.strip()
            i = i.split(',')
            i = np.array(i)
            #i = i.astype(np.float)
            # print(i)
            # print(i.shape)
            # print(type(i))
            # print(i.argmax())
            if i.argmax() == 0:
                train_normal += 1
            else:
                train_attack += 1
        print(f'train_normal: {train_normal} \tPercentage: {train_normal / (train_normal + train_attack)}')
        print(f'train_attack: {train_attack} \tPercentage: {train_attack / (train_normal + train_attack)}')
        print(f'train_total: {train_normal + train_attack}')
        print('train data complete')

def calc_train_labels():
    with open ('train_labels.csv', 'r') as train_file:
        train_data = train_file.readlines()
        # use argmax to find the index position with the max value in the array
        # if the index position is 0 then it is a normal output
        # if the index position is not 0 then it is an attack output
        train_normal = 0
        train_attack = 0
        for i in train_data:
            i = i.strip()
            i = i.split(',')
            i = np.array(i)
            #i = i.astype(np.float)
            # print(i)
            # print(i.shape)
            # print(type(i))
            # print(i.argmax())
            if i.argmax() == 0:
                train_normal += 1
            else:
                train_attack += 1
        print(f'train_normal: {train_normal} \tPercentage: {train_normal / (train_normal + train_attack)}')
        print(f'train_attack: {train_attack} \tPercentage: {train_attack / (train_normal + train_attack)}')
        print(f'train_total: {train_normal + train_attack}')
        print('train data complete')

def calc_total_dataset():
    with open ('kddcup.data.txt', 'r') as total_file:
        total_data = total_file.readlines()
        # use argmax to find the index position with the max value in the array
        # if the index position is 0 then it is a normal output
        # if the index position is not 0 then it is an attack output
        total_normal = 0
        total_attack = 0
        for i in total_data:
            i = i.strip()
            i = i.split(',')
            i = np.array(i)
            #i = i.astype(np.float)
            # print(i)
            # print(i.shape)
            # print(type(i))
            # print(i.argmax())
            if i.argmax() == 0:
                total_normal += 1
            else:
                total_attack += 1
        print(f'total_normal: {total_normal}')
        print(f'total_attack: {total_attack}')
        print(f'total_total: {total_normal + total_attack}')
        print(f'total_normal_percentage: {total_normal / (total_normal + total_attack)}')
        print(f'total_attack_percentage: {total_attack / (total_normal + total_attack)}')
        print('total data complete')

    # calculations for total dataset
    classifications = ['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune', 'smurf', 'guess_passwd', 'pod', 'teardrop', 'portsweep', 'ipsweep', 'land', 'ftp_write', 'back', 'imap', 'satan', 'phf', 'nmap', 'multihop', 'warezmaster', 'warezclient', 'spy', 'rootkit']
    # convert the classifications to a dictionary
    # the key is the classification and the value is intialized to 0
    classifications_dict = {}
    for i in range(len(classifications)):
        classifications_dict[classifications[i]] = 0
    #print(classifications_dict)

    # open kddcup.data.txt and count the number of each classification
    # increment the value of the classification in the dictionary
    with open ('kddcup.data.txt', 'r') as k_file:
        k_data = k_file.readlines()
        for i in k_data:
            for classifications in classifications_dict:
                if classifications in i:
                    classifications_dict[classifications] += 1
    print(classifications_dict)
    print(f'Total Normal Count: {classifications_dict["normal"]} \tPercentage of total: {classifications_dict["normal"] / sum(classifications_dict.values())}')
    print(f'Total Attack Count: {sum(classifications_dict.values()) - classifications_dict["normal"]} /t Percentage of total: {(sum(classifications_dict.values()) - classifications_dict["normal"]) / sum(classifications_dict.values())}')
    print(f'Total Count: {sum(classifications_dict.values())} \tPercentage of total: {sum(classifications_dict.values()) / sum(classifications_dict.values())}')


def main():
    #calc_test_data()
    calc_test_labesl()
    #calc_train_data()
    #calc_train_labels()
    #calc_total_dataset()

if __name__ == "__main__":
    main()