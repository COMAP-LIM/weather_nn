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
    good_lines_subset = random.sample(good_lines, n + 500)
    bad_lines_subset = random.sample(bad_lines, n)

    for i in range(len(good_lines_subset)):
        good_lines_subset[i] = good_lines_subset[i][:-2] + '    0\n'
    
    for i in range(len(bad_lines_subset)):
        bad_lines_subset[i] = bad_lines_subset[i][:-2] + '    1\n'

    n_tot = len(good_lines_subset) + len(bad_lines_subset)

    # Ensuring that there are the same amount of bad and good data in the testing samples.
    # Don't want to use more than 25% of the bad data as testing data, since we 
    # have limited bad data. 
    n_test_samples = int(len(bad_lines_subset)*0.25)
    
    # If 25% of the bad data is more than 12.5% of all the data, use 12.5% of all data as 
    # the number of testing samples
    if n_test_samples > n_tot*0.125:
        n_test_samples = n_tot*0.125
    
    random.shuffle(good_lines_subset)
    random.shuffle(bad_lines_subset)
    testing_set = good_lines_subset[:n_test_samples] + bad_lines_subset[:n_test_samples]

    training_set = good_lines_subset[n_test_samples:] + bad_lines_subset[n_test_samples:]
    
    random.shuffle(training_set)
    random.shuffle(testing_set)

    f1 = open('data/' + filename_train, 'w')
    f2 = open('data/' + filename_test, 'w')
    for i in range(len(training_set)):
        f1.write(training_set[i])

    for i in range(len(testing_set)):
        f2.write(testing_set[i])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python split_train_test.py filename_train filename_test')
        sys.exit()
    filename_train = sys.argv[1]
    filename_test = sys.argv[2]
    split_train_test(filename_train,  filename_test)
