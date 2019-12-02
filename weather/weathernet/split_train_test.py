import random
import sys

def split_train_test(filename_train, filename_test):
    print("Splitting dataset into subsets for training and testing. The training data is written to '%s' and the testing data to %s" %(filename_train, filename_test))
    
    good_obsids_file = 'data/good_subsequences_ALL.txt' 
    bad_obsids_file = 'data/bad_subsequences_ALL.txt'

    f1 = open(good_obsids_file, 'r')
    good_lines = f1.readlines()
    
    f2 = open(bad_obsids_file, 'r')
    bad_lines = f2.readlines()

    n = len(bad_lines)

    # Choose n random files    
    good_lines_subset = random.sample(good_lines, n)
    bad_lines_subset = random.sample(bad_lines, n)

    for i in range(n):
        good_lines_subset[i] = good_lines_subset[i][:-2] + '    0\n'
        bad_lines_subset[i] = bad_lines_subset[i][:-2] + '    1\n'

    all_lines = good_lines_subset + bad_lines_subset
    random.shuffle(all_lines)

    training_set = all_lines[:int(0.75*len(all_lines))]
    testing_set = all_lines[int(0.75*len(all_lines)):]


    f1 = open('data/' + filename_train, 'w')
    f2 = open('data/' + filename_test, 'w')
    for i in range(len(training_set)):
        f1.write(training_set[i])

    for i in range(len(testing_set)):
        f2.write(testing_set[i])

if __name__ == '__main__':
    filename_train = sys.argv[1]
    filename_test = sys.argv[2]
    split_test_train(filename_train,  filename_test)
